import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n = 20
    n_colors = 20
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)

    # Precompute all 4-cycles as tuples of 4 edge indices
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

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
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # For fast checking: for each edge, list of 4-cycles containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # Edge degree in 4-cycle hypergraph (for greedy ordering)
    edge_degree = [len(edge_cycles[ei]) for ei in range(num_edges)]

    # State:
    # edges_set: set of edge indices present in the graph
    # colors: dict mapping edge index to color

    def can_add_edge(ei, c, edges_set, colors):
        """Check if adding edge ei with color c is valid."""
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            # Check if the other 3 edges are present
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [c]
                if len(set(cs)) != 4:
                    return False
        return True

    def try_add_edge(ei, edges_set, colors):
        """Try to add edge ei with any valid color. Returns True if added."""
        valid_colors = list(range(n_colors))
        random.shuffle(valid_colors)
        for c in valid_colors:
            if can_add_edge(ei, c, edges_set, colors):
                edges_set.add(ei)
                colors[ei] = c
                return True
        return False

    def smart_repair(ei, edges_set, colors):
        """
        Attempt to add edge ei by recoloring a small set of existing edges.
        Uses a backtracking search (B&B-like) on the colors of conflicting edges.
        """
        # Identify conflicting cycles
        conflicting_cycles = []
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                conflicting_cycles.append(cyc)
        
        if not conflicting_cycles:
            return try_add_edge(ei, edges_set, colors)

        # Collect all edges involved in these cycles (excluding ei)
        involved_edges = set()
        for cyc in conflicting_cycles:
            for e2 in cyc:
                if e2 != ei:
                    involved_edges.add(e2)
        
        # Limit the scope of repair to avoid explosion
        if len(involved_edges) > 10:
            return False

        involved_edges_list = list(involved_edges)
        random.shuffle(involved_edges_list)
        
        # Save original colors
        original_colors = {e: colors[e] for e in involved_edges_list}
        
        # Domains for each involved edge: all colors except current? 
        # Actually, we can try any color, but we must ensure global validity for ALL cycles involving these edges.
        
        def is_valid_assignment(assignment):
            """
            Check if the assignment (dict edge->color) for involved_edges_list + ei
            results in a valid coloring for all 4-cycles that are fully present 
            and involve at least one edge from involved_edges_list or ei.
            """
            # Create a temporary view of colors
            temp_colors = colors.copy()
            for e, c in assignment.items():
                temp_colors[e] = c
            
            # Check all 4-cycles that contain ei OR any edge in involved_edges_list
            # We only need to check cycles that are FULLY present in edges_set U {ei}
            checked_cycles = set()
            
            # 1. Check cycles containing ei
            for ci in edge_cycles[ei]:
                cyc = all_4cycles[ci]
                if all(e2 in edges_set or e2 == ei for e2 in cyc):
                    # All edges present
                    cs = []
                    for e2 in cyc:
                        if e2 == ei:
                            cs.append(assignment.get(ei, -1)) # ei is in assignment
                        else:
                            cs.append(temp_colors.get(e2, -1))
                    if len(set(cs)) != 4:
                        return False
                    checked_cycles.add(ci)

            # 2. Check cycles containing any involved edge (that don't contain ei, to avoid duplication, 
            #    but we must check if they are fully present)
            for e_inv in involved_edges_list:
                for ci in edge_cycles[e_inv]:
                    if ci in checked_cycles:
                        continue
                    cyc = all_4cycles[ci]
                    # Check if this cycle is fully present in the NEW graph (edges_set + ei)
                    # Note: involved edges are already in edges_set. ei is being added.
                    # If the cycle doesn't contain ei, it was already present and valid.
                    # Changing colors of involved edges might break it.
                    if all(e2 in edges_set for e2 in cyc):
                        cs = [temp_colors.get(e2, -1) for e2 in cyc]
                        if len(set(cs)) != 4:
                            return False
                        checked_cycles.add(ci)
            
            return True

        # Backtracking search
        # We need to assign colors to involved_edges_list and ei.
        # Order: ei first? Or involved edges?
        # Let's try to assign colors to involved edges, then ei.
        
        def backtrack(idx, current_assignment):
            if time.time() > deadline - 2.0:
                return None
            
            if idx == len(involved_edges_list):
                # Try to assign a color to ei
                for c in range(n_colors):
                    current_assignment[ei] = c
                    if is_valid_assignment(current_assignment):
                        return current_assignment.copy()
                return None
            
            e = involved_edges_list[idx]
            # Try colors for e
            # Heuristic: try colors that appear in conflicting cycles first?
            # Or just random.
            possible_colors = list(range(n_colors))
            random.shuffle(possible_colors)
            
            for c in possible_colors:
                current_assignment[e] = c
                # Pruning: Check if this partial assignment already causes a conflict 
                # in any cycle that is now "fully determined" or "almost determined"?
                # For simplicity, we check validity only at the end or with a light check.
                # Light check: If e is in a cycle with 3 other edges whose colors are fixed (in current_assignment or original),
                # and the set of 4 colors has duplicates, prune.
                
                valid_so_far = True
                for ci in edge_cycles[e]:
                    cyc = all_4cycles[ci]
                    # Check if all other edges in this cycle have assigned colors 
                    # (either in current_assignment or are not in involved_edges_list, so they have original colors)
                    # And if the cycle is fully present in edges_set (or with ei? no, ei is separate)
                    # Actually, we only care about cycles that are fully present in the FINAL graph.
                    # Since we are only modifying involved_edges, and ei is added,
                    # we must ensure cycles containing ei are valid (checked later)
                    # and cycles containing e (but not ei) are valid.
                    
                    # If e is in a cycle that does NOT contain ei, and all edges in that cycle are in edges_set,
                    # we should check it now to prune.
                    if ei not in cyc:
                        if all(e2 in edges_set for e2 in cyc):
                            cs = []
                            for e2 in cyc:
                                if e2 in current_assignment:
                                    cs.append(current_assignment[e2])
                                else:
                                    cs.append(colors.get(e2, -1))
                            if len(set(cs)) != 4:
                                valid_so_far = False
                                break
                
                if valid_so_far:
                    res = backtrack(idx + 1, current_assignment)
                    if res is not None:
                        return res
                
                # Unassign
                if e in current_assignment:
                    del current_assignment[e]
            
            return None

        solution = backtrack(0, {})
        
        if solution is not None:
            # Apply solution
            for e, c in solution.items():
                if e == ei:
                    edges_set.add(ei)
                    colors[ei] = c
                else:
                    colors[e] = c
            return True
        else:
            # Restore original colors
            for e, c in original_colors.items():
                colors[e] = c
            return False

    def greedy_build():
        """Greedy edge addition with edge ordering biased towards high-degree edges."""
        edges_set = set()
        colors = {}
        remaining = list(range(num_edges))
        # Sort by degree with some randomness to break ties
        random.shuffle(remaining)
        remaining.sort(key=lambda ei: -edge_degree[ei] + random.uniform(-5, 5))

        for ei in remaining:
            if time.time() > deadline:
                break
            try_add_edge(ei, edges_set, colors)
        return edges_set, colors

    def local_search(edges_set, colors):
        """
        Local search: try to add missing edges.
        If an edge cannot be added, try to re-color existing edges to make it possible.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle missing again to avoid bias
            random.shuffle(missing)

            for ei in missing:
                if time.time() > deadline - 2.0:
                    break
                if ei in edges_set:
                    continue

                # First, try simple addition
                if try_add_edge(ei, edges_set, colors):
                    improved = True
                    missing.remove(ei)
                    continue

                # If simple addition failed, try smart repair
                if smart_repair(ei, edges_set, colors):
                    improved = True
                    missing.remove(ei)
                    continue

    def edge_swap_pass(edges_set, colors):
        """
        Try to remove an edge and add different edges to escape local optima.
        """
        if time.time() > deadline - 3.0:
            return edges_set, colors

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            present = list(edges_set)
            random.shuffle(present)

            for ei_remove in present[:10]:
                if time.time() > deadline - 2.0:
                    break

                removed_color = colors[ei_remove]
                edges_set.remove(ei_remove)
                del colors[ei_remove]

                missing = [ei for ei in range(num_edges) if ei not in edges_set]
                random.shuffle(missing)

                added_any = False
                for ei_add in missing[:20]:
                    if time.time() > deadline - 2.0:
                        break
                    if try_add_edge(ei_add, edges_set, colors):
                        added_any = True

                if added_any:
                    improved = True
                else:
                    edges_set.add(ei_remove)
                    colors[ei_remove] = removed_color

                if improved:
                    break

        return edges_set, colors

    # Run multiple greedy restarts to find a good starting point
    best_edges_set = set()
    best_colors = {}
    best_count = 0

    # Phase 1: Greedy Builds
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        edges_set, colors = greedy_build()
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges_set = edges_set.copy()
            best_colors = colors.copy()

    # Phase 2: Local Search on the best solution
    if time.time() < deadline - 5.0:
        local_search(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 2.5: Edge swap pass
    if time.time() < deadline - 4.0:
        edge_swap_pass(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 3: More restarts with local search if time permits
    if time.time() < deadline - 5.0:
        for r in range(50):
            if time.time() > deadline - 2.0:
                break
            edges_set, colors = greedy_build()
            # Quick local search on this new candidate
            local_search(edges_set, colors)
            if len(edges_set) > best_count:
                best_count = len(edges_set)
                best_edges_set = edges_set.copy()
                best_colors = colors.copy()

    # Final polish: run local search and edge swap on best solution
    if time.time() < deadline - 2.0:
        local_search(best_edges_set, best_colors)
        edge_swap_pass(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Build result
    result = []
    for ei in best_edges_set:
        u, v = all_edges[ei]
        c = best_colors[ei]
        result.append([u, v, c])

    return result
