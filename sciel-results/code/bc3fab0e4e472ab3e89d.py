import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    def get_shortest_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden vertices (except u, v)."""
        if u == v:
            return [u]
        if (u, v) in edge_set:
            return [u, v]
        
        # BFS
        visited = {u}
        queue = deque([(u, [u])])
        while queue:
            node, path = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == v:
                    return path + [v]
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def try_construct_subdivision(branches):
        """Try to find internally disjoint paths for all pairs in branches."""
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        used_internal = set()
        paths = {}
        
        # Identify missing edges
        missing_edges = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = branches[i], branches[j]
                if (u, v) not in edge_set:
                    missing_edges.append((u, v))
        
        # Sort missing edges by shortest path length to prioritize "easy" connections
        # Note: This heuristic might fail if a "long" path is actually required to be disjoint,
        # but it's a good starting point.
        
        # For backtracking, we need to try paths.
        # Given small n, we can use a simple recursive backtracker.
        
        def backtrack(idx, used_internal, paths):
            if idx == len(missing_edges):
                return True
            
            u, v = missing_edges[idx]
            # Find candidate paths
            # We want paths that avoid used_internal and branch_set (except u,v)
            forbidden = used_internal.union(branch_set)
            # u and v are allowed as endpoints
            # BFS for shortest path first
            path = get_shortest_path(u, v, forbidden)
            
            if path is None:
                return False
            
            # Try this path
            internal_nodes = set(path[1:-1])
            if internal_nodes.isdisjoint(used_internal):
                new_used = used_internal.union(internal_nodes)
                paths[(u, v)] = path
                paths[(v, u)] = path[::-1]
                
                if backtrack(idx + 1, new_used, paths):
                    return True
                
                # Backtrack
                del paths[(u, v)]
                del paths[(v, u)]
            
            # If shortest path fails, try other paths?
            # For now, let's just try the shortest. If it fails, we return False.
            # To improve, we could enumerate more paths, but it's complex.
            # Given the constraints, let's assume shortest is best.
            # If we need more robustness, we can add a loop over next shortest paths.
            
            return False

        if backtrack(0, used_internal, paths):
            # Convert paths to list of lists for return
            # Keys are (u,v) where u<v in branches? No, missing_edges has specific order.
            # We need to return paths for ALL pairs.
            # Pairs that are edges: path is [u, v]
            # Pairs that are missing: path is in paths dict
            result_paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = branches[i], branches[j]
                    if (u, v) in edge_set:
                        result_paths.append([u, v])
                    else:
                        # Find the path in paths dict
                        # It might be stored as (u,v) or (v,u)
                        if (u, v) in paths:
                            result_paths.append(paths[(u, v)])
                        elif (v, u) in paths:
                            result_paths.append(paths[(v, u)])
                        else:
                            return None # Should not happen
            return {'branches': list(branches), 'paths': result_paths}
        else:
            return None

    # 1. Find Max Clique
    def find_max_clique():
        # Simple backtracking Bron-Kerbosch or similar
        # For n=24, we can just check all subsets? No, 2^24 is large.
        # Use a simple greedy + local search or exact for small n.
        # Let's use a simple recursive max clique finder.
        
        def expand(clique, candidates):
            if not candidates:
                return clique
            
            # Pick a vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            
            # Option 1: Include v
            new_candidates = candidates & adj[v]
            res1 = expand(clique + [v], new_candidates)
            
            # Option 2: Exclude v
            res2 = expand(clique, candidates - {v})
            
            return res1 if len(res1) > len(res2) else res2
        
        return expand([], set(range(n)))

    best_clique = find_max_clique()
    best_result = None
    if len(best_clique) >= 2:
        # Construct subdivision for the clique (all paths are edges)
        branches = best_clique
        paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                paths.append([branches[i], branches[j]])
        best_result = {'branches': branches, 'paths': paths}
    
    # 2. Try to find larger subdivisions
    # Start from len(best_clique) + 1 up to 10
    # If best_clique is small, we might jump.
    start_k = len(best_clique) + 1
    if start_k < 3: start_k = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0
    
    for k in range(start_k, 11):
        if time.time() > deadline:
            break
            
        # Iterate over combinations
        # Optimization: Only consider subsets with enough internal edges
        # Required edges m >= k + k*(k-1)/2 - n
        req_edges = k + k*(k-1)//2 - n
        
        for comb in combinations(range(n), k):
            if time.time() > deadline:
                break
                
            # Check internal edges
            internal_edges = 0
            for i in range(k):
                for j in range(i+1, k):
                    if (comb[i], comb[j]) in edge_set:
                        internal_edges += 1
            
            if internal_edges < req_edges:
                continue
            
            # Try to construct
            res = try_construct_subdivision(list(comb))
            if res:
                # Found a valid subdivision with k branches
                # Since we iterate k increasing, the first one found for this k is valid.
                # We want the LARGEST k.
                # So we should break and return this?
                # Or continue to see if k+1 is possible?
                # The loop goes up to 10. If we find k=8, we should check k=9, 10.
                # So we store it and continue.
                if best_result is None or k > len(best_result['branches']):
                    best_result = res
        
    # Fallback if no result (should not happen as clique >= 2)
    if best_result is None:
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
        
    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result

# EVOLVE-BLOCK-END
