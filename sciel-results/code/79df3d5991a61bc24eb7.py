"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n = 20
    n_colors = 20
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    # Precompute all 4-cycles as tuples of 4 edges (as frozenset of edge indices for fast lookup)
    # A 4-cycle is a set of 4 edges forming a cycle of length 4
    # We represent each edge by its index in all_edges
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    # Find all 4-cycles: for each pair of vertices (a,b) and (c,d) that are non-adjacent in the cycle
    # A 4-cycle has vertices a-b-c-d-a. The edges are (a,b), (b,c), (c,d), (d,a).
    # Alternatively, think of it as two diagonals: (a,c) and (b,d) are NOT edges of the cycle.
    # We'll enumerate by choosing 4 distinct vertices and checking the 3 possible 4-cycles on them.
    # For 4 vertices w, x, y, z, the 3 possible 4-cycles are:
    #   w-x-y-z-w: edges (w,x),(x,y),(y,z),(z,w)
    #   w-x-z-y-w: edges (w,x),(x,z),(z,y),(y,w)
    #   w-y-x-z-w: edges (w,y),(y,x),(x,z),(z,w)

    from itertools import combinations
    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        # 3 distinct 4-cycles
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            # Convert to edge indices
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # For fast checking: for each edge, list of 4-cycles containing it
    edge_cycles = [[] for _ in range(len(all_edges))]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    best_edges = []
    best_colors = {}

    def check_validity(edges_set, colors):
        """Check if the coloring is valid: every 4-cycle in the graph has 4 distinct colors."""
        # Only check 4-cycles that are fully present in the graph
        for ci, cyc in enumerate(all_4cycles):
            if all(ei in edges_set for ei in cyc):
                cs = [colors[ei] for ei in cyc]
                if len(set(cs)) != 4:
                    return False
        return True

    def greedy_build():
        """Greedy edge addition with random edge ordering and random color assignment."""
        edges_set = set()
        colors = {}
        remaining = list(range(len(all_edges)))
        random.shuffle(remaining)

        for ei in remaining:
            if time.time() > deadline:
                break
            u, v = all_edges[ei]
            # Try to find a color that doesn't create a bad 4-cycle
            valid_colors = list(range(n_colors))
            random.shuffle(valid_colors)

            for c in valid_colors:
                # Tentatively add edge (ei, c) and check if any 4-cycle containing ei becomes invalid
                ok = True
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei):
                        # This 4-cycle would be complete
                        cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [c]
                        if len(set(cs)) != 4:
                            ok = False
                            break
                if ok:
                    edges_set.add(ei)
                    colors[ei] = c
                    break

        return len(edges_set), edges_set, colors

    # Run multiple greedy restarts
    best_count = 0
    best_eset = set()
    best_col = {}

    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        count, eset, col = greedy_build()
        if count > best_count:
            best_count = count
            best_eset = eset
            best_col = col

    # Local search: try to add more edges to the best solution found
    if time.time() < deadline - 5.0:
        # Try adding edges not in best_eset
        missing = [ei for ei in range(len(all_edges)) if ei not in best_eset]
        random.shuffle(missing)
        for ei in missing:
            if time.time() > deadline - 3.0:
                break
            if ei in best_eset:
                continue
            valid_colors = list(range(n_colors))
            random.shuffle(valid_colors)
            for c in valid_colors:
                ok = True
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in best_eset for e2 in cyc if e2 != ei):
                        cs = [best_col.get(e2) for e2 in cyc if e2 != ei] + [c]
                        if len(set(cs)) != 4:
                            ok = False
                            break
                if ok:
                    best_eset.add(ei)
                    best_col[ei] = c
                    break

    # Also try some more restarts if time permits
    if time.time() < deadline - 10.0:
        for r in range(100):
            if time.time() > deadline - 3.0:
                break
            count, eset, col = greedy_build()
            if count > best_count:
                best_count = count
                best_eset = eset
                best_col = col
                # Local search on this new best
                if time.time() < deadline - 5.0:
                    missing = [ei for ei in range(len(all_edges)) if ei not in best_eset]
                    random.shuffle(missing)
                    for ei in missing:
                        if time.time() > deadline - 3.0:
                            break
                        if ei in best_eset:
                            continue
                        valid_colors = list(range(n_colors))
                        random.shuffle(valid_colors)
                        for c in valid_colors:
                            ok = True
                            for ci in edge_cycles[ei]:
                                cyc = all_4cycles[ci]
                                if all(e2 in best_eset for e2 in cyc if e2 != ei):
                                    cs = [best_col.get(e2) for e2 in cyc if e2 != ei] + [c]
                                    if len(set(cs)) != 4:
                                        ok = False
                                        break
                            if ok:
                                best_eset.add(ei)
                                best_col[ei] = c
                                break

    # Build result
    result = []
    for ei in best_eset:
        u, v = all_edges[ei]
        c = best_col[ei]
        result.append([u, v, c])

    return result
# EVOLVE-BLOCK-END
