"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    # Find maximal cliques
    def bron_kerbosch(R, P_set, X, cliques):
        if time.time() > deadline:
            return
        if not P_set and not X:
            cliques.append(list(R))
            return
        # Pivot selection
        if P_set or X:
            pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], cliques)
            P_set.remove(v)
            X.add(v)

    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques)
    cliques.sort(key=len, reverse=True)

    best_k = 0
    best_branches = None
    best_paths = None

    def is_valid_clique_subdivision(branches):
        """
        Checks if the given set of branches forms a valid clique subdivision.
        Returns (True, paths_dict) or (False, None).
        Uses backtracking to find disjoint paths.
        """
        if time.time() > deadline:
            return False, None
        
        k = len(branches)
        if k < 2:
            return True, {}
        
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Heuristic: Sort pairs by difficulty.
        # Difficulty: Non-adjacent pairs are harder. 
        # Among non-adjacent, pairs with fewer common neighbors (excluding branches) are harder.
        def pair_difficulty(pair):
            u, v = pair
            if v in adj[u]:
                return 0 # Easiest: direct edge
            # Count available interior vertices
            common = adj[u] & adj[v]
            common -= branch_set
            # Also subtract vertices that might be used? Hard to know. 
            # Just use number of common neighbors as a proxy for "ease".
            # Fewer common neighbors = harder.
            return -len(common)
            
        pairs.sort(key=pair_difficulty)
        
        # Backtracking solver
        # State: idx (index in pairs), used_interior (set of vertices used in interiors)
        # We need to find a path for pairs[idx] that avoids used_interior and branch_set - {u, v}
        
        def solve(idx, used_interior, current_paths):
            if time.time() > deadline:
                return False, None
            
            if idx == len(pairs):
                return True, current_paths
            
            u, v = pairs[idx]
            
            # If u and v are adjacent, path is [u, v], no interior used
            if v in adj[u]:
                path = [u, v]
                return solve(idx + 1, used_interior, current_paths + [(u, v, path)])
            
            # Find candidate paths from u to v
            # BFS to find shortest paths first? Or just DFS?
            # Let's generate simple paths avoiding forbidden vertices.
            forbidden = used_interior | (branch_set - {u, v})
            
            # We want to try paths that use "less critical" vertices?
            # For now, try shortest paths first.
            
            paths_found = []
            # BFS to find shortest path length
            dist = {u: 0}
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v:
                    break
                for nxt in adj[curr]:
                    if nxt in forbidden:
                        continue
                    if nxt not in dist:
                        dist[nxt] = dist[curr] + 1
                        q.append(nxt)
            
            if v not in dist:
                return False, None
            
            min_len = dist[v]
            
            # DFS to find all simple paths of length min_len (and maybe min_len+1)
            # To avoid explosion, limit number of paths we try.
            # Actually, just try one path at a time in a backtracking manner.
            
            def find_path_dfs(curr, target, path, visited):
                if time.time() > deadline:
                    return None
                if curr == target:
                    return list(path)
                for nxt in adj[curr]:
                    if nxt in forbidden:
                        continue
                    if nxt in visited:
                        continue
                    if nxt == target:
                        # Check if path length is acceptable? 
                        # If we just want *any* path, take it.
                        # But shorter is better for disjointness.
                        path.append(nxt)
                        res = list(path)
                        path.pop()
                        return res
                    path.append(nxt)
                    visited.add(nxt)
                    res = find_path_dfs(nxt, target, path, visited)
                    if res:
                        return res
                    visited.remove(nxt)
                    path.pop()
                return None

            # Try to find a path
            # To improve success rate, we might want to try multiple paths if the first one fails later in the recursion.
            # This requires generating multiple paths.
            # Let's generate a few candidate paths (e.g., up to 5 shortest ones).
            
            candidate_paths = []
            # Use a modified DFS to find up to 5 simple paths
            def find_k_paths(curr, target, path, visited, k_limit):
                if time.time() > deadline:
                    return []
                if curr == target:
                    return [list(path)]
                results = []
                for nxt in adj[curr]:
                    if nxt in forbidden:
                        continue
                    if nxt in visited:
                        continue
                    path.append(nxt)
                    visited.add(nxt)
                    res = find_k_paths(nxt, target, path, visited, k_limit)
                    results.extend(res)
                    if len(results) >= k_limit:
                        break
                    visited.remove(nxt)
                    path.pop()
                return results

            # Get up to 10 candidate paths
            candidates = find_k_paths(u, v, [u], {u}, 10)
            
            # Sort candidates by length
            candidates.sort(key=len)
            
            for path in candidates:
                internal = set(path[1:-1])
                if not (internal & used_interior):
                    new_used = used_interior | internal
                    res, res_paths = solve(idx + 1, new_used, current_paths + [(u, v, path)])
                    if res:
                        return True, res_paths
            
            return False, None

        valid, paths_dict = solve(0, set(), [])
        if valid:
            # Convert paths_dict to list of paths
            result_paths = [p[2] for p in paths_dict]
            return True, result_paths
        return False, None

    # Strategy:
    # 1. Start with the largest clique.
    # 2. Try to extend it by adding vertices one by one.
    # 3. If extension fails, try the next largest clique.
    
    # Also, try to find a better starting point by checking if we can add vertices to the max clique.
    
    # To be thorough, we can try all maximal cliques, but prioritize larger ones.
    # Since N=24, the number of maximal cliques is manageable.
    
    for clique in cliques:
        if time.time() > deadline - 5:
            break
            
        current_branches = set(clique)
        current_k = len(current_branches)
        
        if current_k <= best_k:
            continue
            
        # Check if the clique itself is valid (it always is)
        valid, paths = is_valid_clique_subdivision(list(current_branches))
        if valid:
            if current_k > best_k:
                best_k = current_k
                best_branches = list(current_branches)
                best_paths = paths
        
        # Try to extend
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Sort candidate vertices by degree (high degree first? or low?)
            # High degree vertices are more likely to have disjoint paths.
            candidates = [v for v in range(n) if v not in current_branches]
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                candidate_set = current_branches | {v}
                valid, paths = is_valid_clique_subdivision(list(candidate_set))
                if valid:
                    current_branches = candidate_set
                    current_k += 1
                    if current_k > best_k:
                        best_k = current_k
                        best_branches = list(current_branches)
                        best_paths = paths
                    improved = True
                    break # Restart the loop with the new set
        
    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
        
    # Ensure types are correct
    return {'branches': list(best_branches), 'paths': [list(p) for p in best_paths]}

# EVOLVE-BLOCK-END
