import time
import sys
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def is_clique(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if (S_list[i], S_list[j]) not in edge_set:
                    return False
        return True

    def find_max_clique():
        # Bron-Kerbosch with pivot
        best = []
        
        def expand(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = R[:]
                return
            
            # Pivot choice: vertex in P union X with max degree in P
            if P:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                pivot = None
            
            candidates = list(P)
            if pivot is not None:
                candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                expand(R + [v], P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        expand([], set(range(n)), set())
        return best

    max_clique = find_max_clique()
    best_S = set(max_clique)
    best_k = len(best_S)
    
    def verify_subdivision(S):
        k = len(S)
        if k < 2:
            return False, None
        
        if is_clique(S):
            S_list = list(S)
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    paths.append([S_list[i], S_list[j]])
            return True, {'branches': S_list, 'paths': paths}
        
        S_list = list(S)
        S_set = set(S_list)
        # Identify non-edge pairs
        non_edge_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                u, v = S_list[i], S_list[j]
                if (u, v) not in edge_set:
                    non_edge_pairs.append((u, v))
        
        # Necessary condition: enough internal vertices
        if len(non_edge_pairs) > n - k:
            return False, None
            
        # Find disjoint paths for non_edge_pairs
        # We use a backtracking search
        # State: used_internal set, index in non_edge_pairs
        used_internal = set()
        paths_found = {}
        
        # Sort pairs by difficulty (shortest path length first)
        def get_shortest_path_len(u, v, forbidden):
            # BFS
            visited = {u}
            queue = deque([(u, 0)])
            while queue:
                node, dist = queue.popleft()
                if node == v:
                    return dist
                for neighbor in adj[node]:
                    if neighbor == v:
                        return dist + 1
                    if neighbor not in visited and neighbor not in forbidden and neighbor not in S_set:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return float('inf')

        # Pre-compute initial distances for sorting
        # Note: forbidden is empty initially
        dists = [(get_shortest_path_len(u, v, set()), u, v) for u, v in non_edge_pairs]
        dists.sort(key=lambda x: x[0])
        sorted_pairs = [(u, v) for _, u, v in dists]
        
        def backtrack(idx):
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            forbidden = used_internal.union(S_set - {u, v})
            
            # Find all simple paths from u to v avoiding forbidden
            # Limit path length to 4 or 5 to avoid explosion
            # Since n is small, we can try to find all paths up to length 4
            found_paths = []
            
            def dfs(node, path):
                if len(path) > 5: # Max length heuristic
                    return
                if node == v:
                    found_paths.append(path[:])
                    return
                for neighbor in adj[node]:
                    if neighbor == v:
                        found_paths.append(path + [v])
                        return
                    if neighbor not in path and neighbor not in forbidden and neighbor not in S_set:
                        dfs(neighbor, path + [neighbor])
            
            dfs(u, [u])
            
            # If no paths found, fail
            if not found_paths:
                return False
            
            # Try each path
            # Sort by length (shortest first)
            found_paths.sort(key=len)
            
            for path in found_paths:
                internal_nodes = set(path[1:-1])
                if internal_nodes.isdisjoint(used_internal):
                    used_internal.update(internal_nodes)
                    paths_found[(u, v)] = path
                    if backtrack(idx + 1):
                        return True
                    used_internal.difference_update(internal_nodes)
                    del paths_found[(u, v)]
            
            return False

        if backtrack(0):
            res_paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    u, v = S_list[i], S_list[j]
                    if (u, v) in edge_set:
                        res_paths.append([u, v])
                    else:
                        # Find the path in paths_found
                        # The key might be (v, u) or (u, v)
                        if (u, v) in paths_found:
                            res_paths.append(paths_found[(u, v)])
                        elif (v, u) in paths_found:
                            res_paths.append(paths_found[(v, u)][::-1])
                        else:
                            # Should not happen
                            return False, None
            return True, {'branches': S_list, 'paths': res_paths}
        else:
            return False, None

    # Search for larger sets
    start_time = time.time()
    deadline = start_time + 85.0
    
    degrees = [len(adj[i]) for i in range(n)]
    vertices_sorted = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    current_k = best_k
    
    while current_k < n and time.time() < deadline:
        current_k += 1
        found = False
        
        # Strategy A: Extend best_S
        if len(best_S) == current_k - 1:
            outside = [v for v in range(n) if v not in best_S]
            # Sort by number of neighbors in best_S
            outside.sort(key=lambda x: len(adj[x] & best_S), reverse=True)
            for v in outside:
                if time.time() > deadline:
                    break
                candidate_S = best_S.union({v})
                valid, res = verify_subdivision(candidate_S)
                if valid:
                    best_S = candidate_S
                    best_k = current_k
                    found = True
                    break
        
        # Strategy B: Random sampling
        if not found:
            pool_size = min(n, 15)
            pool = vertices_sorted[:pool_size]
            if current_k > pool_size:
                pool = vertices_sorted[:min(n, current_k + 5)]
            
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts and time.time() < deadline:
                candidate_S = set(random.sample(pool, current_k))
                valid, res = verify_subdivision(candidate_S)
                if valid:
                    best_S = candidate_S
                    best_k = current_k
                    found = True
                    break
                attempts += 1
        
        if not found:
            break

    valid, res = verify_subdivision(best_S)
    if valid:
        result = res
    else:
        result = {'branches': list(max_clique), 'paths': [[max_clique[i], max_clique[j]] for i in range(len(max_clique)) for j in range(i+1, len(max_clique))]}
        
    result['branches'] = list(result['branches'])
    result['paths'] = [list(p) for p in result['paths']]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
