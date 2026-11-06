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

    def greedy_build():
        """Greedy edge addition with random edge ordering."""
        edges_set = set()
        colors = {}
        remaining = list(range(num_edges))
        random.shuffle(remaining)

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
                
                # If simple addition failed, try to fix conflicts by re-coloring
                # Identify conflicting cycles
                conflicting_cycles = []
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei):
                        # This cycle is complete except for ei
                        # Check if it's a conflict
                        # We need to find if there's a color for ei that works.
                        # Actually, can_add_edge already checked all colors.
                        # So we know no color works.
                        # We need to change a color of one of the other 3 edges.
                        conflicting_cycles.append(cyc)
                
                if not conflicting_cycles:
                    continue

                # Try to re-color one of the edges in the conflicting cycles
                # Pick a random conflicting cycle
                random.shuffle(conflicting_cycles)
                target_cyc = conflicting_cycles[0]
                # The other 3 edges in this cycle
                other_edges = [e2 for e2 in target_cyc if e2 != ei]
                random.shuffle(other_edges)
                
                for e_change in other_edges:
                    old_color = colors[e_change]
                    # Try changing e_change to a different color
                    possible_colors = [c for c in range(n_colors) if c != old_color]
                    random.shuffle(possible_colors)
                    
                    for new_c in possible_colors:
                        # Check if changing e_change to new_c is valid for all cycles containing e_change
                        # AND if it allows ei to be added
                        valid = True
                        # 1. Check if e_change can be recolored
                        for ci in edge_cycles[e_change]:
                            cyc = all_4cycles[ci]
                            # Get colors of other edges in this cycle
                            # Note: ei is not in edges_set yet, so if ei is in this cycle, 
                            # the cycle is not complete, so no constraint violation from this cycle yet.
                            # But if we add ei later, it must be valid.
                            # For now, we just need to ensure the recoloring doesn't break existing complete cycles.
                            if all(e2 in edges_set for e2 in cyc):
                                cs = [colors.get(e2) for e2 in cyc if e2 != e_change] + [new_c]
                                if len(set(cs)) != 4:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # 2. Check if this recoloring allows ei to be added
                        # We need to check if there exists a color for ei now.
                        # Temporarily set colors[e_change] = new_c
                        colors[e_change] = new_c
                        
                        added = False
                        valid_colors_for_ei = list(range(n_colors))
                        random.shuffle(valid_colors_for_ei)
                        for c_ei in valid_colors_for_ei:
                            if can_add_edge(ei, c_ei, edges_set, colors):
                                edges_set.add(ei)
                                colors[ei] = c_ei
                                added = True
                                break
                        
                        if added:
                            improved = True
                            missing.remove(ei)
                            break
                        else:
                            # Revert color change
                            colors[e_change] = old_color
                            
                    if improved:
                        break

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

    # Build result
    result = []
    for ei in best_edges_set:
        u, v = all_edges[ei]
        c = best_colors[ei]
        result.append([u, v, c])

    return result
