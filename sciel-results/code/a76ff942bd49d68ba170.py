"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Precompute shortest paths
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # Max Clique via Bron-Kerbosch
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Check if a set B has a valid subdivision using CSP
    def check_subdivision_csp(B, time_limit=5.0):
        B_set = set(B)
        k = len(B)
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        # Get all short paths for each pair
        def get_all_short_paths(u, v, max_len=4):
            paths = []
            def dfs(curr, end, path, visited):
                if len(path) - 1 > max_len:
                    return
                if curr == end:
                    paths.append(list(path))
                    return
                for nxt in adj_list[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        path.append(nxt)
                        dfs(nxt, end, path, visited)
                        path.pop()
                        visited.remove(nxt)
            dfs(u, v, [u], {u})
            return paths
        
        domains = {}
        for u, v in non_adj_pairs:
            paths = get_all_short_paths(u, v, max_len=4)
            valid_paths = []
            for p in paths:
                interior = p[1:-1]
                if any(w in B_set for w in interior):
                    continue
                valid_paths.append(p)
            if not valid_paths:
                return False, {}
            domains[(u, v)] = valid_paths
        
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        used_interiors = set()
        
        def backtrack(idx, assignment):
            if time.time() > time_limit:
                return None
            if idx == len(sorted_pairs):
                return assignment
            
            u, v = sorted_pairs[idx]
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used_interiors for w in interior):
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                result = backtrack(idx + 1, assignment)
                if result is not None:
                    return result
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return None
        
        result = backtrack(0, {})
        return (result is not None), result

    best_k = len(best_clique)
    best_B = list(best_clique)
    best_paths = {}

    # Try to find larger sets via random sampling
    import random
    random.seed(42)
    
    for k in range(best_k + 1, min(n + 1, best_k + 6)):
        if time.time() > deadline - 10:
            break
        
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline - 10:
            B = random.sample(range(n), k)
            ok, paths_dict = check_subdivision_csp(B, time_limit=3.0)
            if ok:
                best_k = k
                best_B = B
                best_paths = paths_dict
                break
            attempts += 1

    # Construct final result
    B = best_B
    k = len(B)
    paths = []
    
    for i in range(k):
        for j in range(i+1, k):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths.append([u, v])
            else:
                if (u, v) in best_paths:
                    paths.append(best_paths[(u, v)])
                elif (v, u) in best_paths:
                    paths.append(list(reversed(best_paths[(v, u)])))
                else:
                    p = get_path(u, v)
                    if p:
                        paths.append(p)

    return {'branches': list(B), 'paths': paths}

# EVOLVE-BLOCK-END
