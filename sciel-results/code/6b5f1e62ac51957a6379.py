"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # Precompute all 4-cycles for verification
    all_cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        all_cycles.append((a, b, c, d))
        all_cycles.append((a, b, d, c))
        all_cycles.append((a, c, b, d))

    def verify(edge_dict):
        """Check all 4-cycles have 4 distinct colors."""
        for cycle in all_cycles:
            a, b, c, d = cycle
            e1 = edge_dict.get((min(a,b), max(a,b)))
            e2 = edge_dict.get((min(b,c), max(b,c)))
            e3 = edge_dict.get((min(c,d), max(c,d)))
            e4 = edge_dict.get((min(d,a), max(d,a)))
            if e1 is None or e2 is None or e3 is None or e4 is None:
                continue
            if len({e1, e2, e3, e4}) != 4:
                return False, cycle
        return True, None

    # === STRUCTURED CONSTRUCTION: Algebraic edge coloring ===
    # Key insight: Color edge (u,v) by (v - u) mod n.
    # For a 4-cycle a-b-c-d-a with all vertices distinct:
    #   Colors: (b-a)%n, (c-b)%n, (d-c)%n, (a-d)%n
    #   Sum = 0 mod n, and no individual color is 0 (since all vertices distinct).
    #   If any two colors were equal, e.g. (b-a) ≡ (c-b) mod n,
    #   then b ≡ (a+c)/2 mod n, meaning b is the midpoint of a and c.
    #   This CAN happen, so pure modular coloring isn't always valid.
    #
    # Strategy: Use a combination of modular structure + greedy repair.
    # Start with a structured coloring, then fix conflicts by edge removal or recoloring.

    # Approach A: Try multiple algebraic colorings and pick the best valid one.
    # Approach B: Greedy edge addition with constraint checking.

    # Let's try a hybrid: structured initial coloring + local repair via recoloring.
    # If recoloring fails, remove the problematic edge.

    def structured_coloring(seed_offset=0):
        """Generate a structured coloring using modular arithmetic with offset."""
        edge_dict = {}
        for u in range(n):
            for v in range(u + 1, n):
                # Multiple strategies for color assignment
                color = (v - u + seed_offset) % num_colors
                edge_dict[(u, v)] = color
        return edge_dict

    def greedy_repair(edge_dict, max_steps=50000):
        """
        Iteratively fix 4-cycle color conflicts by recoloring or removing edges.
        Returns (edge_dict, num_edges_removed, is_valid).
        """
        removed_edges = set()
        step = 0
        while step < max_steps:
            if time.time() > DEADLINE:
                break
            step += 1

            valid, conflict_cycle = verify(edge_dict)
            if valid:
                return edge_dict, len(removed_edges), True

            if conflict_cycle is None:
                # No conflict found but verify said invalid? Shouldn't happen.
                break

            a, b, c, d = conflict_cycle
            e1_key = (min(a,b), max(a,b))
            e2_key = (min(b,c), max(b,c))
            e3_key = (min(c,d), max(c,d))
            e4_key = (min(d,a), max(d,a))

            # Check which edges are still present
            present_keys = [k for k in [e1_key, e2_key, e3_key, e4_key]
                           if k in edge_dict and k not in removed_edges]

            if len(present_keys) < 4:
                # Not a full cycle anymore (some edge removed)
                # Remove one more to be safe, or skip
                continue

            # Get the 4 colors
            colors_in_cycle = [edge_dict[k] for k in present_keys]

            # Try to recolor one of the edges to fix the conflict
            fixed = False
            for idx, key in enumerate(present_keys):
                old_color = edge_dict[key]
                # Try each other color
                for new_color in range(num_colors):
                    if new_color == old_color:
                        continue
                    # Check if this recoloring creates new conflicts
                    # Quick check: just try it and see if it helps
                    edge_dict[key] = new_color
                    valid_test, _ = verify(edge_dict)
                    if valid_test:
                        fixed = True
                        break
                    # Also check if it reduces total conflicts (heuristic)
                    edge_dict[key] = old_color

                if fixed:
                    break

            if not fixed:
                # Remove the edge that appears in the most 4-cycles with conflicts
                # Heuristic: remove a random edge from the cycle
                key_to_remove = present_keys[0]
                # Actually, let's try each edge removal and pick the one that helps most
                best_key = None
                best_score = -1
                for key in present_keys:
                    del edge_dict[key]
                    # Count remaining violations (approximate: just check this cycle is gone)
                    # Quick heuristic: remove and check if graph is still "dense"
                    remaining = len(edge_dict)
                    if remaining > best_score:
                        best_score = remaining
                        best_key = key
                    edge_dict[key] = old_color  # restore

                if best_key is not None:
                    del edge_dict[best_key]
                    removed_edges.add(best_key)

        valid, _ = verify(edge_dict)
        return edge_dict, len(removed_edges), valid

    # Try multiple structured colorings with different offsets
    best_result = None
    best_edge_count = 0

    for offset in range(num_colors):
        if time.time() > DEADLINE - 5:
            break

        edge_dict = structured_coloring(offset)
        edge_dict, removed, is_valid = greedy_repair(edge_dict)

        if is_valid:
            edge_count = len(edge_dict)
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_result = edge_dict.copy()

    # If structured approach didn't yield a good result, try pure greedy construction
    if best_result is None or best_edge_count < 100:
        if time.time() > DEADLINE - 10:
            # Fallback to star graph (guaranteed valid, 19 edges)
            return [[0, v, 0] for v in range(1, n)]

        # Greedy construction: add edges one by one, choosing color to avoid conflicts
        edge_dict = {}
        # Order edges by some heuristic (e.g., by vertex degree potential)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

        for (u, v) in all_edges:
            if time.time() > DEADLINE - 2:
                break

            # Find a valid color for this edge
            valid_color = None
            for color in range(num_colors):
                # Check all 4-cycles that would include this edge
                # A 4-cycle containing (u,v): u-w-x-v-u or u-x-w-v-u
                conflict = False
                for w in range(n):
                    if w == u or w == v:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        # Cycle: u-w-x-v-u
                        ew = (min(u,w), max(u,w))
                        wx = (min(w,x), max(w,x))
                        xv = (min(x,v), max(x,v))
                        if ew in edge_dict and wx in edge_dict and xv in edge_dict:
                            c1 = edge_dict[ew]
                            c2 = edge_dict[wx]
                            c3 = edge_dict[xv]
                            if color in (c1, c2, c3):
                                conflict = True
                                break
                        # Cycle: u-x-w-v-u
                        ux = (min(u,x), max(u,x))
                        xw = (min(x,w), max(x,w))
                        wv = (min(w,v), max(w,v))
                        if ux in edge_dict and xw in edge_dict and wv in edge_dict:
                            c1 = edge_dict[ux]
                            c2 = edge_dict[xw]
                            c3 = edge_dict[wv]
                            if color in (c1, c2, c3):
                                conflict = True
                                break
                    if conflict:
                        break
                if not conflict:
                    valid_color = color
                    break

            if valid_color is not None:
                edge_dict[(u, v)] = valid_color

        if len(edge_dict) > best_edge_count:
            best_result = edge_dict
            best_edge_count = len(edge_dict)

    # Final verification and output
    if best_result is not None and len(best_result) > 0:
        valid, _ = verify(best_result)
        if valid:
            edges_out = []
            for (u, v), color in best_result.items():
                edges_out.append([u, v, color])
            return edges_out

    # Ultimate fallback: star graph (19 edges, guaranteed valid)
    return [[0, v, 0] for v in range(1, n)]
# EVOLVE-BLOCK-END
