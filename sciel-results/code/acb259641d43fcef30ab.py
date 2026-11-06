import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 20
    n_colors = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # Precompute all 4-cycles
    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
            all_4cycles.append(edge_idxs)
    
    num_cycles = len(all_4cycles)
    
    # For each edge, which cycles contain it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)
    
    # State arrays for fast access
    in_graph = [False] * num_edges
    color = [-1] * num_edges
    
    def can_add_edge(ei, c):
        """Check if adding edge ei with color c maintains validity."""
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            # Check if other 3 edges are present
            if all(in_graph[e2] for e2 in cyc if e2 != ei):
                cs = set()
                for e2 in cyc:
                    if e2 == ei:
                        cs.add(c)
                    else:
                        cs.add(color[e2])
                if len(cs) != 4:
                    return False
        return True
    
    def try_add_edge(ei):
        """Try to add edge ei with any valid color."""
        for c in range(n_colors):
            if can_add_edge(ei, c):
                in_graph[ei] = True
                color[ei] = c
                return True
        return False
    
    def remove_edge(ei):
        in_graph[ei] = False
        color[ei] = -1
    
    def greedy_build(seed=None):
        """Greedy edge addition with random ordering."""
        if seed is not None:
            random.seed(seed)
        
        in_graph[:] = [False] * num_edges
        color[:] = [-1] * num_edges
        
        remaining = list(range(num_edges))
        random.shuffle(remaining)
        
        for ei in remaining:
            if time.time() > deadline:
                break
            try_add_edge(ei)
        
        return sum(in_graph)
    
    def local_search():
        """
        Local search: try to add missing edges, with recoloring when stuck.
        """
        missing = [ei for ei in range(num_edges) if not in_graph[ei]]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            random.shuffle(missing)
            
            for ei in missing:
                if time.time() > deadline - 3.0:
                    break
                if in_graph[ei]:
                    continue
                
                # Try simple addition
                if try_add_edge(ei):
                    improved = True
                    missing.remove(ei)
                    continue
                
                # Try recoloring to enable addition
                # Find conflicting cycles
                conflicting_cycles = []
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(in_graph[e2] for e2 in cyc if e2 != ei):
                        conflicting_cycles.append(cyc)
                
                if not conflicting_cycles:
                    continue
                
                # Try recoloring edges in conflicting cycles
                random.shuffle(conflicting_cycles)
                for target_cyc in conflicting_cycles[:3]:
                    if time.time() > deadline - 3.0:
                        break
                    
                    other_edges = [e2 for e2 in target_cyc if e2 != ei]
                    random.shuffle(other_edges)
                    
                    for e_change in other_edges:
                        old_color = color[e_change]
                        possible_colors = [c for c in range(n_colors) if c != old_color]
                        random.shuffle(possible_colors)
                        
                        for new_c in possible_colors[:8]:
                            # Check if recoloring is valid for existing complete cycles
                            valid = True
                            for ci in edge_cycles[e_change]:
                                cyc = all_4cycles[ci]
                                if all(in_graph[e2] for e2 in cyc):
                                    cs = set()
                                    for e2 in cyc:
                                        if e2 == e_change:
                                            cs.add(new_c)
                                        else:
                                            cs.add(color[e2])
                                    if len(cs) != 4:
                                        valid = False
                                        break
                            
                            if not valid:
                                continue
                            
                            # Temporarily recolor
                            color[e_change] = new_c
                            
                            # Try to add ei
                            added = False
                            for c_ei in range(n_colors):
                                if can_add_edge(ei, c_ei):
                                    in_graph[ei] = True
                                    color[ei] = c_ei
                                    added = True
                                    break
                            
                            if added:
                                improved = True
                                missing.remove(ei)
                                break
                            else:
                                color[e_change] = old_color
                        
                        if improved:
                            break
                
                if improved:
                    break
        
        return sum(in_graph)
    
    # Phase 1: Multiple greedy restarts
    best_count = 0
    best_in_graph = [False] * num_edges
    best_color = [-1] * num_edges
    
    num_restarts = min(80, int((deadline - start_time) / 0.5))
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        in_graph[:] = [False] * num_edges
        color[:] = [-1] * num_edges
        
        remaining = list(range(num_edges))
        random.shuffle(remaining)
        
        for ei in remaining:
            if time.time() > deadline - 15.0:
                break
            try_add_edge(ei)
        
        count = sum(in_graph)
        if count > best_count:
            best_count = count
            best_in_graph = in_graph[:]
            best_color = color[:]
    
    # Phase 2: Local search on best solution
    in_graph = best_in_graph[:]
    color = best_color[:]
    
    if time.time() < deadline - 5.0:
        local_search()
        count = sum(in_graph)
        if count > best_count:
            best_count = count
            best_in_graph = in_graph[:]
            best_color = color[:]
    
    # Phase 3: More restarts with local search
    if time.time() < deadline - 5.0:
        for r in range(30):
            if time.time() > deadline - 2.0:
                break
            
            in_graph[:] = [False] * num_edges
            color[:] = [-1] * num_edges
            
            remaining = list(range(num_edges))
            random.shuffle(remaining)
            
            for ei in remaining:
                if time.time() > deadline - 2.0:
                    break
                try_add_edge(ei)
            
            count = sum(in_graph)
            
            # Quick local search
            if time.time() < deadline - 2.0:
                missing = [ei for ei in range(num_edges) if not in_graph[ei]]
                random.shuffle(missing)
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    random.shuffle(missing)
                    for ei in missing:
                        if time.time() > deadline - 2.0:
                            break
                        if in_graph[ei]:
                            continue
                        if try_add_edge(ei):
                            improved = True
                            missing.remove(ei)
                            break
                count = sum(in_graph)
            
            if count > best_count:
                best_count = count
                best_in_graph = in_graph[:]
                best_color = color[:]
    
    # Build result
    result = []
    for ei in range(num_edges):
        if best_in_graph[ei]:
            u, v = all_edges[ei]
            c = best_color[ei]
            result.append([u, v, c])
    
    return result
