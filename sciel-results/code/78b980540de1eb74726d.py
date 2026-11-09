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
                # We only need to consider vertices in P that are NOT neighbors of pivot
                # If a vertex is a neighbor of pivot, it is "covered" by pivot
                candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                expand(R + [v], P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        expand([], set(range(n)), set())
        return best

    # Step 1: Find Max Clique
    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_S = set(max_clique)
    
    # Helper to verify if a set S is a valid branch set
    def verify_subdivision(S):
        k = len(S)
        if k < 2:
            return False, None
        
        # If it's a clique, paths are just edges
        if is_clique(S):
            S_list = list(S)
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    paths.append([S_list[i], S_list[j]])
            return True, {'branches': S_list, 'paths': paths}
        
        # Otherwise, we need to find disjoint paths
        # This is expensive, so we use a heuristic backtracker
        S_list = list(S)
        S_set = set(S_list)
        pairs = [(S_list[i], S_list[j]) for i in range(k) for j in range(i + 1, k)]
        
        # Sort pairs by difficulty: no edge first, then fewer common neighbors
        def pair_difficulty(p):
            u, v = p
            if (u, v) in edge_set:
                return 0
            # Count common neighbors not in S
            common = (adj[u] & adj[v]) - S_set
            return -len(common)
            
        pairs.sort(key=pair_difficulty)
        
        used_internal = set()
        paths_found = {}
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forbidden = used_internal.union(S_set - {u, v})
            
            # Find a path from u to v avoiding forbidden
            # Use BFS to find shortest path
            visited = {u}
            queue = deque([(u, [u])])
            found_paths = []
            
            # We might want multiple options if the first one fails later,
            # but for now, let's just grab the shortest one. 
            # To improve, we could collect a few shortest paths.
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found_paths.append(path)
                    break
                for neighbor in adj[node]:
                    if neighbor == v:
                        found_paths.append(path + [v])
                        break
                    if neighbor not in visited and neighbor not in forbidden:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
                if found_paths:
                    break
            
            if not found_paths:
                return False
            
            # Try the found path
            path = found_paths[0]
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
            for u, v in pairs:
                res_paths.append(paths_found[(u, v)])
            return True, {'branches': S_list, 'paths': res_paths}
        else:
            return False, None

    # Step 2: Try to find larger k
    # Start from best_k + 1
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Pool of candidates: all vertices, sorted by degree
    degrees = [len(adj[i]) for i in range(n)]
    vertices_sorted = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will try to construct sets of size k = best_k + 1, best_k + 2, etc.
    # Strategy: 
    # 1. Take the max clique and try to add vertices one by one.
    # 2. Random sampling from top-degree vertices.
    
    found_improvement = True
    current_k = best_k
    
    while current_k < n and time.time() < deadline:
        current_k += 1
        found = False
        
        # Strategy A: Extend the best known set
        # Try adding one vertex from outside best_S to best_S
        if len(best_S) == current_k - 1:
            outside = [v for v in range(n) if v not in best_S]
            # Sort outside by degree
            outside.sort(key=lambda x: degrees[x], reverse=True)
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
        
        # Strategy B: Random sampling from top vertices
        if not found:
            # Sample from top 15 vertices
            pool_size = min(n, 15)
            pool = vertices_sorted[:pool_size]
            
            # If current_k > pool_size, we can't sample only from pool
            # We must include some lower degree vertices or expand pool
            if current_k > pool_size:
                pool = vertices_sorted[:min(n, current_k + 5)]
            
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time.time() < deadline:
                # Random sample of size current_k
                candidate_S = set(random.sample(pool, current_k))
                valid, res = verify_subdivision(candidate_S)
                if valid:
                    best_S = candidate_S
                    best_k = current_k
                    found = True
                    break
                attempts += 1
        
        if not found:
            # If we couldn't find a set of size current_k, 
            # it's likely we can't find larger ones easily.
            # But maybe we missed one. 
            # For now, break to save time, as verification is expensive.
            break

    # Final verification of best_S to get the paths
    valid, res = verify_subdivision(best_S)
    if valid:
        result = res
    else:
        # Fallback to max clique (which is always valid)
        result = {'branches': list(max_clique), 'paths': [[max_clique[i], max_clique[j]] for i in range(len(max_clique)) for j in range(i+1, len(max_clique))]}
        
    # Ensure types
    result['branches'] = list(result['branches'])
    result['paths'] = [list(p) for p in result['paths']]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
