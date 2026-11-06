"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices.

Strategy:
  1. Build a fast conflict structure: for every pair of edges that can share a colour
     and appear together in some 4-cycle, we record the conflict.
  2. Use a branch-and-bound / backtracking search over edges (in a good order) to
     maximize the number of edges while keeping every 4-cycle rainbow.
  3. The search uses a greedy colouring: when adding an edge, pick the colour that
     minimizes future conflicts. If no colour works, backtrack.
  4. We also try multiple different edge orderings and colour heuristics to escape
     local optima.
  5. A final local-improvement phase (swap colours, add back removed edges) polishes.
"""
import time
import random
from itertools import combinations

N = 20
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s wall-clock kill

def _all_4_cycles():
    """Generate all 4-cycles as frozensets of 4 edges (u,v) with u<v."""
    cycles = []
    for a, b, c, d in combinations(range(N), 4):
        for perm in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
            x, y, z, w = perm
            edges = frozenset((min(u, v), max(u, v))
                              for u, v in [(x, y), (y, z), (z, w), (w, x)])
            cycles.append(edges)
    return cycles

def _build_conflict_map(cycles):
    """
    For each pair of edges (e1, e2) that appear together in at least one 4-cycle,
    record that they cannot share a colour.
    Returns: conflict_map[e1] = set of e2 that conflict with e1 (i.e., share a 4-cycle).
    """
    conflict_map = {e: set() for e in [(i, j) for i in range(N) for j in range(i+1, N)]}
    for cyc in cycles:
        edges = list(cyc)
        for i in range(4):
            for j in range(i+1, 4):
                conflict_map[edges[i]].add(edges[j])
                conflict_map[edges[j]].add(edges[i])
    return conflict_map

def _build_cycle_lookup(cycles):
    """
    For each edge, list the cycles that contain it.
    """
    edge_to_cycles = {}
    for idx, cyc in enumerate(cycles):
        for e in cyc:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)
    return edge_to_cycles

def _verify_result(result, cycles):
    """Verify a result is valid. Returns True if valid."""
    if not result:
        return False
    edges_set = set()
    colors = {}
    for u, v, c in result:
        e = (min(u, v), max(u, v))
        if e in edges_set:
            return False
        edges_set.add(e)
        colors[e] = c
    for cyc in cycles:
        if not all(e in edges_set for e in cyc):
            continue
        cols = [colors[e] for e in cyc]
        if len(set(cols)) < 4:
            return False
    return True

def _backtrack_search(edge_order, conflict_map, edge_to_cycles, cycles, deadline, max_edges_target):
    """
    Backtracking search: process edges in edge_order. For each edge, try to assign
    a colour such that no 4-cycle containing this edge has a colour conflict.
    If we can't assign a colour, skip the edge (it's not in the final graph).
    Track the best (most edges) valid partial assignment.
    
    We use a forward-checking style: maintain for each colour used in a 4-cycle,
    which edges have that colour.
    """
    t_start = time.time()
    n_edges = len(edge_order)
    
    # State:
    # assigned[edge_index] = colour or -1 if not assigned
    # edge_colour[edge] = colour
    # For each 4-cycle, track colours used: cycle_colours[cycle_idx] = set of colours
    # Actually, for efficiency, track per-cycle which colours are "taken"
    
    # Better approach: for each edge we try to add, check all 4-cycles containing it.
    # A 4-cycle is bad if among its present edges, two share a colour.
    
    # Let's use a different formulation:
    # We decide for each edge: include or not, and if include, what colour.
    # Constraint: for every 4-cycle, all included edges in it must have distinct colours.
    
    # Forward checking: when we assign colour c to edge e, for every 4-cycle containing e,
    # no other edge in that cycle can use colour c.
    
    # State representation:
    # edge_color: dict edge -> colour (only for included edges)
    # banned: for each (edge, colour) pair that's forbidden, track why.
    # Simpler: for each edge, maintain a set of colours that are still available.
    
    # Let's do: avail[edge] = set of colours that can be assigned to edge
    # Initially all colours 0..N-1 available for every edge.
    # When we assign colour c to edge e, for every edge e' in a 4-cycle with e,
    # remove c from avail[e'].
    
    # This is essentially arc consistency / forward checking.
    
    all_edges_set = set(edge_order)
    avail = {e: set(range(N)) for e in all_edges_set}
    edge_color = {}
    included = []
    best_included = []
    best_count = 0
    
    # Process edges in order
    for idx, e in enumerate(edge_order):
        if time.time() - t_start > deadline:
            break
        
        # Try each colour for edge e
        assigned = False
        # Sort colours by how few conflicts they cause (fewer bans on other edges)
        # For simplicity, try colours in order 0..N-1, but prefer those that ban fewer
        colour_options = sorted(avail[e], key=lambda c: sum(
            1 for e2 in conflict_map[e] if e2 in avail and c in avail[e2]
        ))
        
        for c in colour_options:
            # Check: can we assign c to e?
            # We need to verify that for every 4-cycle containing e,
            # no other already-assigned edge in that cycle has colour c.
            valid = True
            for cyc_idx in edge_to_cycles.get(e, []):
                cyc = cycles[cyc_idx]
                for e2 in cyc:
                    if e2 == e:
                        continue
                    if e2 in edge_color and edge_color[e2] == c:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Assign c to e
            edge_color[e] = c
            included.append(e)
            assigned = True
            
            # Forward check: for every edge e2 that shares a 4-cycle with e,
            # remove colour c from avail[e2]
            for e2 in conflict_map[e]:
                if e2 in avail:
                    avail[e2].discard(c)
            
            # Check for any edge that now has empty avail (and is in the future)
            # This is a simplification - we don't do full propagation
            break
        
        if not assigned:
            # Skip this edge
            pass
        else:
            # Update best if we have more edges
            if len(included) > best_count:
                best_count = len(included)
                best_included = list(included)
                best_colors = dict(edge_color)
    
    # Build result from best
    if best_included:
        result = []
        for e in best_included:
            u, v = e
            c = best_colors.get(e, 0)
            result.append([u, v, c])
        return result
    return []

def _local_improvement(result, cycles, conflict_map, deadline):
    """
    Try to improve the result by:
    1. Trying to add back edges that were skipped.
    2. Swapping colours to free up options.
    """
    t_start = time.time()
    if not result:
        return result
    
    edges_set = set()
    colors = {}
    for u, v, c in result:
        e = (min(u, v), max(u, v))
        edges_set.add(e)
        colors[e] = c
    
    all_possible_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    missing_edges = [e for e in all_possible_edges if e not in edges_set]
    
    # Try to add missing edges one by one
    for e in missing_edges:
        if time.time() - t_start > deadline:
            break
        if e in edges_set:
            continue
        
        # Find a valid colour for e
        for c in range(N):
            # Check all 4-cycles containing e
            valid = True
            for cyc in cycles:
                if e not in cyc:
                    continue
                for e2 in cyc:
                    if e2 == e:
                        continue
                    if e2 in edges_set and colors[e2] == c:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                edges_set.add(e)
                colors[e] = c
                result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges_set)]
                break
    
    # Try colour swaps to enable more edges
    for _ in range(50):  # limited number of swap attempts
        if time.time() - t_start > deadline:
            break
        
        # Find a missing edge and try to find a swap that enables it
        missing = [e for e in all_possible_edges if e not in edges_set]
        if not missing:
            break
        
        # Pick a random missing edge
        e_target = random.choice(missing)
        
        # For each colour c, check if swapping some edge's colour to c would work
        for c in range(N):
            # Check if c is blocked by some edge in a 4-cycle with e_target
            blocking_edges = []
            for cyc in cycles:
                if e_target not in cyc:
                    continue
                for e2 in cyc:
                    if e2 == e_target:
                        continue
                    if e2 in edges_set and colors[e2] == c:
                        blocking_edges.append(e2)
            
            if not blocking_edges:
                # We can just add e_target with colour c
                edges_set.add(e_target)
                colors[e_target] = c
                result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges_set)]
                break
            
            # Try swapping one blocking edge to a different colour
            for e_block in blocking_edges:
                # Find a colour for e_block that doesn't conflict
                for c2 in range(N):
                    if c2 == colors[e_block]:
                        continue
                    # Check if c2 is valid for e_block
                    valid = True
                    for cyc in cycles:
                        if e_block not in cyc:
                            continue
                        for e3 in cyc:
                            if e3 == e_block:
                                continue
                            if e3 in edges_set and colors[e3] == c2:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Do the swap
                        colors[e_block] = c2
                        edges_set.add(e_target)
                        colors[e_target] = c
                        result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges_set)]
                        break
                if e_target in edges_set:
                    break
            if e_target in edges_set:
                break
    
    return result

def _greedy_construct(edge_order, conflict_map, edge_to_cycles, cycles, deadline):
    """
    Greedy construction: process edges in order, add edge if there's a valid colour.
    """
    t_start = time.time()
    edges_set = set()
    colors = {}
    
    for e in edge_order:
        if time.time() - t_start > deadline:
            break
        if e in edges_set:
            continue
        
        # Find a valid colour
        for c in range(N):
            valid = True
            for cyc_idx in edge_to_cycles.get(e, []):
                cyc = cycles[cyc_idx]
                for e2 in cyc:
                    if e2 == e:
                        continue
                    if e2 in edges_set and colors[e2] == c:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                edges_set.add(e)
                colors[e] = c
                break
    
    result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges_set)]
    return result

def _try_multiple_orderings(cycles, conflict_map, edge_to_cycles, deadline):
    """
    Try multiple edge orderings with greedy construction + local improvement.
    """
    t_start = time.time()
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    best_result = []
    best_count = 0
    
    # Different orderings
    orderings = []
    
    # Natural order
    orderings.append(list(all_edges))
    
    # Random orderings
    rng = random.Random(123)
    for _ in range(20):
        order = list(all_edges)
        rng.shuffle(order)
        orderings.append(order)
    
    # Degree-based orderings (process high-degree edges first or last)
    for _ in range(5):
        order = list(all_edges)
        rng.shuffle(order)
        # Sort by number of conflicts (edges that share 4-cycles)
        order.sort(key=lambda e: len(conflict_map[e]), reverse=True)
        orderings.append(order)
    
    for order in orderings:
        if time.time() - t_start > deadline - 5.0:
            break
        
        result = _greedy_construct(order, conflict_map, edge_to_cycles, cycles, deadline - 5.0)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Local improvement on best
    if best_result:
        improved = _local_improvement(best_result, cycles, conflict_map, deadline - 2.0)
        if len(improved) > best_count:
            best_result = improved
    
    return best_result

def _structured_construction(cycles, conflict_map, edge_to_cycles, deadline):
    """
    Try structured colourings:
    - Colour (u,v) = (u+v) mod N
    - Colour (u,v) = (u-v) mod N  
    - Colour (u,v) = (u*3+v*7) mod N
    etc. Then greedily remove conflicting edges.
    """
    t_start = time.time()
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    best_result = []
    best_count = 0
    
    color_funcs = [
        lambda u, v: (u + v) % N,
        lambda u, v: (u - v) % N,
        lambda u, v: (u + 2*v) % N,
        lambda u, v: (2*u + v) % N,
        lambda u, v: (u * v) % N,
        lambda u, v: (u * 3 + v * 7) % N,
        lambda u, v: (u * 5 + v * 3) % N,
        lambda u, v: (u * 7 + v * 11) % N,
        lambda u, v: (u * 13 + v * 17) % N,
        lambda u, v: (u * 19 + v * 11) % N,
    ]
    
    for cf in color_funcs:
        if time.time() - t_start > deadline - 5.0:
            break
        
        # Start with all edges and the structured colouring
        edges_set = set(all_edges)
        colors = {e: cf(*e) % N for e in all_edges}
        
        # Greedily remove edges that cause conflicts
        changed = True
        while changed:
            if time.time() - t_start > deadline - 5.0:
                break
            changed = False
            
            # Find all conflicting pairs
            for e in list(edges_set):
                c = colors[e]
                # Check if any other edge in a 4-cycle with e has the same colour
                for e2 in conflict_map[e]:
                    if e2 in edges_set and colors[e2] == c:
                        # Conflict: remove one of them
                        # Remove the one that's in fewer cycles (less impact)
                        if len(edge_to_cycles.get(e, [])) <= len(edge_to_cycles.get(e2, [])):
                            edges_set.remove(e)
                            del colors[e]
                        else:
                            edges_set.remove(e2)
                            del colors[e2]
                        changed = True
                        break
        
        result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges_set)]
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Local improvement
    if best_result:
        improved = _local_improvement(best_result, cycles, conflict_map, deadline - 2.0)
        if len(improved) > best_count:
            best_result = improved
    
    return best_result

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    cycles = _all_4_cycles()
    conflict_map = _build_conflict_map(cycles)
    edge_to_cycles = _build_cycle_lookup(cycles)
    
    best_result = []
    best_count = 0
    
    # Method 1: Structured construction
    if deadline - time.time() > 20.0:
        result = _structured_construction(cycles, conflict_map, edge_to_cycles, deadline - 20.0)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Method 2: Multiple orderings with greedy + local improvement
    if deadline - time.time() > 30.0:
        result = _try_multiple_orderings(cycles, conflict_map, edge_to_cycles, deadline - 30.0)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Method 3: Backtracking search on a subset of edges
    if deadline - time.time() > 40.0:
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        # Use a good ordering: sort by conflict degree
        edge_order = sorted(all_edges, key=lambda e: len(conflict_map[e]), reverse=True)
        
        result = _backtrack_search(edge_order, conflict_map, edge_to_cycles, cycles,
                                   deadline - 40.0, len(all_edges))
        if result:
            improved = _local_improvement(result, cycles, conflict_map, deadline - 10.0)
            if len(improved) > best_count:
                best_count = len(improved)
                best_result = improved
    
    # Final verification
    if best_result and _verify_result(best_result, cycles):
        return best_result
    
    # Fallback: star graph
    fallback = [[0, v, v] for v in range(1, N)]
    return fallback
