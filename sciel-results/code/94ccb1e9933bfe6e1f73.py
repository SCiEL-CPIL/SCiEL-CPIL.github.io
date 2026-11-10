# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    """
    Decompose each instance graph into edge-disjoint simple paths covering all edges.
    Uses Cycle Removal and Splicing heuristic.
    """
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycle(adj, edges_set):
        """Find a cycle in the graph defined by adj and edges_set."""
        visited = set()
        parent = {}
        
        def dfs(u):
            visited.add(u)
            for v in adj[u]:
                if (min(u,v), max(u,v)) not in edges_set:
                    continue
                if v not in visited:
                    parent[v] = u
                    res = dfs(v)
                    if res:
                        return res
                elif v != parent.get(u, -1):
                    # Cycle found
                    cycle = [u, v]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        cycle.append(curr)
                    return cycle
            return None
        
        for u in adj:
            if u not in visited:
                res = dfs(u)
                if res:
                    return res
        return None

    def decompose_into_paths(n, edges):
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # 1. Remove all cycles
        remaining_edges = set(edges)
        cycles = []
        
        while True:
            cycle = find_cycle(adj, remaining_edges)
            if not cycle:
                break
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1)%len(cycle)]
                e = (min(u,v), max(u,v))
                remaining_edges.discard(e)
            cycles.append(cycle)
            
        # 2. Extract paths from remaining forest
        paths = []
        # Find vertices with degree 1 or 0 in remaining_edges
        deg = defaultdict(int)
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Start from leaves
        for u in range(n):
            if deg[u] == 1 and u not in [p[0] for p in paths]: # Simple check, might need better tracking
                # Walk the path
                path = [u]
                prev = -1
                curr = u
                while True:
                    # Find next
                    nexts = [v for v in adj[curr] if (min(curr,v), max(curr,v)) in remaining_edges and v != prev]
                    if not nexts:
                        break
                    nxt = nexts[0]
                    path.append(nxt)
                    prev, curr = curr, nxt
                if len(path) > 1:
                    paths.append(path)
        
        # 3. Break cycles into paths and store info for splicing
        # We need to map cycle to its "broken path" representation
        cycle_infos = []
        for c in cycles:
            # c is a list of vertices
            # Remove one edge to make it a path
            # Let's remove the last edge (c[-1], c[0])
            break_edge = (min(c[-1], c[0]), max(c[-1], c[0]))
            c_path = c[:] # c[0]...c[-1]
            # The path is c[0] -> c[1] -> ... -> c[-1]
            paths.append(c_path)
            cycle_infos.append({
                'cycle': c,
                'path': c_path,
                'break_edge': break_edge
            })
            
        # 4. Splicing Optimization
        # We maintain a list of paths.
        # We maintain a list of active cycle_infos.
        # A cycle_info is active if its 'path' is still intact in the current paths.
        
        def get_path_id(path):
            # Simple hash based on sorted edges
            es = set()
            for i in range(len(path)-1):
                es.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
            return frozenset(es)
            
        current_paths = [p[:] for p in paths]
        active_cycles = [ci for ci in cycle_infos]
        
        improved = True
        while improved:
            improved = False
            # Try to find a splice
            # For each active cycle
            for ci in active_cycles[:]:
                if not active_cycles: break
                # Check if ci['path'] is still in current_paths
                # We can check by edge set
                ci_path_edges = set()
                for i in range(len(ci['path'])-1):
                    ci_path_edges.add((min(ci['path'][i], ci['path'][i+1]), max(ci['path'][i], ci['path'][i+1])))
                
                # Find the path object in current_paths that matches ci_path_edges
                target_path_idx = -1
                for idx, p in enumerate(current_paths):
                    p_edges = set()
                    for i in range(len(p)-1):
                        p_edges.add((min(p[i], p[i+1]), max(p[i], p[i+1])))
                    if p_edges == ci_path_edges:
                        target_path_idx = idx
                        break
                        
                if target_path_idx == -1:
                    # Cycle path was modified or removed, deactivate this cycle info
                    active_cycles.remove(ci)
                    continue
                
                # Try to splice using this cycle
                # We look for an edge (u, v) in ci['cycle'] (excluding break_edge)
                # such that u is an endpoint of some path P1 and v is an endpoint of some path P2
                # and P1 != P2 (or P1 is the target path? No, if P1 is target path, we break it)
                
                # Get endpoints of all paths
                # path_endpoints: vertex -> list of (path_idx, is_start)
                endpoints = defaultdict(list)
                for idx, p in enumerate(current_paths):
                    if len(p) > 1:
                        endpoints[p[0]].append((idx, True))
                        endpoints[p[-1]].append((idx, False))
                        
                found_splice = False
                # Iterate through edges of the cycle
                for i in range(len(ci['cycle'])):
                    u = ci['cycle'][i]
                    v = ci['cycle'][(i+1)%len(ci['cycle'])]
                    e = (min(u,v), max(u,v))
                    if e == ci['break_edge']:
                        continue
                    
                    # Check if u and v are endpoints
                    if u in endpoints and v in endpoints:
                        # Get candidates
                        # We want P1 ending at u and P2 starting at v
                        # Or P1 ending at v and P2 starting at u
                        
                        # Case 1: P1 ends at u, P2 starts at v
                        p1_candidates = [idx for idx, is_start in endpoints[u] if not is_start]
                        p2_candidates = [idx for idx, is_start in endpoints[v] if is_start]
                        
                        for p1_idx in p1_candidates:
                            for p2_idx in p2_candidates:
                                if p1_idx == p2_idx: continue
                                # Check if P1 or P2 is the target path (ci['path'])
                                # If so, we are breaking the cycle path
                                
                                P1 = current_paths[p1_idx]
                                P2 = current_paths[p2_idx]
                                
                                # Construct new path
                                # P1 ends at u. P2 starts at v.
                                # New path: P1 + [v] + P2[1:]? No.
                                # P1 is [..., u]. P2 is [v, ...].
                                # Edge (u,v) connects them.
                                # New path: P1 + P2[1:]
                                new_path = P1 + P2[1:]
                                
                                # Construct C_rem
                                # C_rem is the cycle minus edge (u,v)
                                # It starts at v and ends at u
                                c_rem = [v]
                                curr = v
                                # Walk cycle from v to u
                                while curr != u:
                                    # Find next in cycle
                                    # Next is the one after curr in cycle order
                                    # Cycle is ci['cycle']
                                    idx_c = ci['cycle'].index(curr)
                                    next_c = ci['cycle'][(idx_c+1)%len(ci['cycle'])]
                                    if (min(curr, next_c), max(curr, next_c)) == (u,v) or (min(u,v), max(u,v)) == (min(curr, next_c), max(curr, next_c)):
                                         # Skip if this is the edge we used?
                                         # Actually, we are walking the cycle. We just need to follow the cycle.
                                         # If we hit u, stop.
                                         pass
                                    c_rem.append(next_c)
                                    curr = next_c
                                    if len(c_rem) > len(ci['cycle']): break # Safety
                                
                                # Verify C_rem is simple and uses correct edges
                                # And that C_rem + new_path covers all edges of P1, P2, and Cycle
                                
                                # Update paths
                                # Remove P1, P2
                                # Remove target_path (ci['path'])
                                # Add new_path, C_rem
                                
                                # Identify indices to remove
                                remove_indices = {p1_idx, p2_idx, target_path_idx}
                                # If p1_idx == target_path_idx or p2_idx == target_path_idx, set handles duplicates
                                
                                new_paths_list = []
                                for idx, p in enumerate(current_paths):
                                    if idx not in remove_indices:
                                        new_paths_list.append(p)
                                
                                new_paths_list.append(new_path)
                                new_paths_list.append(c_rem)
                                
                                current_paths = new_paths_list
                                active_cycles.remove(ci)
                                improved = True
                                found_splice = True
                                break
                        if found_splice: break
                    if found_splice: break
                if found_splice: break
                
        return current_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_into_paths(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
