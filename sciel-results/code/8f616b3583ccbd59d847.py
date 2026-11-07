"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 78  # safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    n = P['n']
    edges_list = P['edges']
    
    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i
    
    # Adjacency with edge indices, sorted for determinism
    adj = [set() for _ in range(n)]
    adj_edges = [[] for _ in range(n)]  # list of (neighbor, edge_idx)
    for idx, (u, v) in enumerate(edges_list):
        adj[u].add(v)
        adj[v].add(u)
        adj_edges[u].append((v, idx))
        adj_edges[v].append((u, idx))
    for v in range(n):
        adj_edges[v].sort()
    
    # Check degree constraints for Hamiltonian cycles
    min_degree = min(len(adj[v]) for v in range(n))
    max_possible_cycles = min(10, min_degree // 2)
    max_possible_cycles = min(max_possible_cycles, len(edges_list) // n)
    
    def find_hamiltonian_cycles(k_target, deadline):
        """Find up to k_target edge-disjoint Hamiltonian cycles using DFS with pruning."""
        used_edges = [False] * len(edges_list)
        cycles = []
        node_count = [0]  # mutable counter for debugging
        
        def dfs_cycle(start, path, visited_mask, depth):
            """Find a Hamiltonian cycle starting at 'start' avoiding used edges."""
            if time.time() > deadline:
                return None
            node_count[0] += 1
            
            if depth == n:
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in edge_index:
                    eidx = edge_index[key]
                    if not used_edges[eidx]:
                        return list(path)
                return None
            
            last = path[-1]
            remaining = n - depth - 1  # vertices left to visit after current
            
            for neighbor, eidx in adj_edges[last]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges[eidx]:
                    continue
                
                # Pruning: ensure neighbor can reach all remaining unvisited vertices
                # Check that for each unvisited vertex (except neighbor), there's a path
                # through unvisited vertices + neighbor using unused edges
                # Simplified: check degree of unvisited vertices in remaining subgraph
                new_mask = visited_mask | (1 << neighbor)
                
                # Quick prune: for each unvisited vertex w, check it has at least 
                # 2 unused edges to unvisited vertices or to the path endpoints
                # This is expensive; use a lighter check
                if remaining > 0:
                    # Check that neighbor has enough unused edges to unvisited vertices
                    # to potentially form a path through all remaining
                    unvisited = [w for w in range(n) if not (new_mask & (1 << w))]
                    if unvisited:
                        # For each unvisited vertex, count unused edges to other unvisited or to start
                        # If any vertex has 0 usable edges, prune
                        for w in unvisited:
                            usable = 0
                            for nb, e2 in adj_edges[w]:
                                if used_edges[e2]:
                                    continue
                                if nb == start or not (new_mask & (1 << nb)):
                                    usable += 1
                            if usable < 2:  # needs at least 2 edges to continue path
                                break
                        else:
                            pass  # all good, continue
                        # If we broke, this neighbor is bad - but the check above doesn't break outer loop
                        # Let me redo this properly
                    
                # Better pruning: check connectivity of unvisited subgraph
                # For simplicity, just check that each unvisited vertex (except those 
                # adjacent to start) has at least one unused edge to another unvisited vertex
                # or to start
                viable = True
                unvisited = [w for w in range(n) if not (new_mask & (1 << w))]
                for w in unvisited:
                    has_edge = False
                    for nb, e2 in adj_edges[w]:
                        if used_edges[e2]:
                            continue
                        if nb == start or not (new_mask & (1 << nb)):
                            has_edge = True
                            break
                    if not has_edge:
                        viable = False
                        break
                
                if not viable:
                    continue
                
                visited_mask_new = new_mask
                path.append(neighbor)
                used_edges[eidx] = True
                result = dfs_cycle(start, path, visited_mask_new, depth + 1)
                used_edges[eidx] = False
                path.pop()
                if result is not None:
                    return result
            return None
        
        # Try to find cycles one by one
        for c in range(k_target):
            if time.time() > deadline - 3:
                break
            
            found = dfs_cycle(0, [0], 1, 1)
            if found is None:
                break
            
            # Mark edges of this cycle as used
            for i in range(n):
                u = found[i]
                v = found[(i + 1) % n]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges[eidx] = True
            
            cycles.append(found)
        
        return cycles

    # Try to find maximum number of edge-disjoint Hamiltonian cycles
    best_cycles = None
    best_count = 0
    
    for k in range(max_possible_cycles, 0, -1):
        if time.time() > deadline - 2:
            break
        
        cycles = find_hamiltonian_cycles(k, deadline)
        
        if cycles is not None and len(cycles) > best_count:
            best_cycles = cycles
            best_count = len(cycles)
            
            if best_count >= max_possible_cycles:
                break
    
    # Fallback: if we didn't find enough, try to find at least 1
    if best_cycles is None or len(best_cycles) == 0:
        for i, (u, v) in enumerate(edges_list):
            if time.time() > deadline:
                break
            for w in adj[u] & adj[v]:
                if w != u and w != v:
                    best_cycles = [[u, v, w]]
                    best_count = 1
                    break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        return [[0, 1, 2]]
    
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
