"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import deque, defaultdict
from itertools import combinations

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
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Precompute all-pairs shortest paths
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
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
    
    # Find all simple paths between u and v with length <= max_len
    # Returns list of paths (each path is a list from u to v)
    def find_paths(u, v, max_len, forbidden):
        """Find simple paths from u to v of length <= max_len avoiding forbidden vertices."""
        paths = []
        if u in forbidden or v in forbidden:
            return paths
        
        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited and nxt not in forbidden:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
        
        dfs(u, [u], {u})
        return paths
    
    # Check if set B has a valid clique subdivision.
    # Returns (True, paths_dict) or (False, None)
    # paths_dict maps (u,v) -> path for non-adjacent pairs
    def check_subdivision(B, max_path_len=6, time_limit=5.0):
        start_t = time.time()
        B_set = set(B)
        k = len(B)
        
        # Identify non-adjacent pairs
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))
        
        if not non_adj_pairs:
            return True, {}
        
        # Generate candidate paths for each non-adjacent pair
        # Sort pairs by number of available paths (MRV)
        domains = {}
        for u, v in non_adj_pairs:
            # Try increasing path lengths
            paths = find_paths(u, v, max_path_len, B_set)
            if not paths:
                return False, None
            domains[(u, v)] = paths
        
        # Sort by domain size (MRV heuristic)
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        
        # Backtracking search with MRV
        used_interiors = set()
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            for path in domains[(u, v)]:
                interior = path[1:-1]
                if any(w in used_interiors for w in interior):
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return False
        
        if backtrack(0):
            return True, dict(assignment)
        return False, None
    
    # Bron-Kerbosch for maximum clique
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
    deadline = start_time + 55.0
    
    # Initialize with max clique
    best_k = len(best_clique)
    best_B = list(best_clique)
    best_paths = {}
    
    # Verify max clique
    ok, paths_dict = check_subdivision(best_B, max_path_len=6, time_limit=5.0)
    if ok:
        best_paths = paths_dict
    else:
        # Max clique not valid, try smaller subsets
        for sz in range(len(best_clique)-1, 1, -1):
            if time.time() > deadline - 20:
                break
            for sub in combinations(best_clique, sz):
                B = list(sub)
                ok, paths_dict = check_subdivision(B, max_path_len=6, time_limit=3.0)
                if ok:
                    best_k = sz
                    best_B = B
                    best_paths = paths_dict
                    break
            if best_k == sz:
                break
    
    # Greedy extension: try to add vertices one at a time
    vertices_by_degree = sorted(range(n), key=lambda v: -degrees[v])
    
    # Multiple greedy strategies
    strategies = [
        'degree_desc',    # Add highest degree first
        'degree_asc',     # Add lowest degree first  
        'random',         # Random order
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 15:
            break
        
        current_B = list(best_B)
        current_paths = dict(best_paths)
        
        # Try to extend
        extended = True
        while extended and time.time() < deadline - 10:
            extended = False
            candidates = [v for v in range(n) if v not in current_B]
            
            if strategy == 'degree_desc':
                candidates.sort(key=lambda v: -degrees[v])
            elif strategy == 'degree_asc':
                candidates.sort(key=lambda v: degrees[v])
            else:
                random.seed(42)
                random.shuffle(candidates)
            
            for v in candidates:
                if time.time() > deadline - 10:
                    break
                
                new_B = current_B + [v]
                ok, new_paths = check_subdivision(new_B, max_path_len=6, time_limit=3.0)
                if ok:
                    current_B = new_B
                    current_paths = new_paths
                    if len(current_B) > best_k:
                        best_k = len(current_B)
                        best_B = list(current_B)
                        best_paths = new_paths
                    extended = True
                    break  # Restart
    
    # Local search: try swapping vertices to improve
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for i in range(len(best_B)):
                if time.time() > deadline - 8:
                    break
                for v in range(n):
                    if v in best_B:
                        continue
                    # Try swapping best_B[i] with v
                    new_B = best_B[:i] + [v] + best_B[i+1:]
                    ok, new_paths = check_subdivision(new_B, max_path_len=6, time_limit=2.0)
                    if ok:
                        best_B = new_B
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
    
    # Random restarts with different seeds
    if time.time() < deadline - 5:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            
            # Start from a random subset of size best_k
            B = random.sample(range(n), best_k)
            ok, paths_dict = check_subdivision(B, max_path_len=6, time_limit=2.0)
            if ok:
                current_B = B
                current_paths = paths_dict
                # Try to extend
                extended = True
                while extended and time.time() < deadline - 3:
                    extended = False
                    candidates = [v for v in range(n) if v not in current_B]
                    random.shuffle(candidates)
                    for v in candidates:
                        if time.time() > deadline - 3:
                            break
                        new_B = current_B + [v]
                        ok, new_paths = check_subdivision(new_B, max_path_len=6, time_limit=1.5)
                        if ok:
                            current_B = new_B
                            current_paths = new_paths
                            if len(current_B) > best_k:
                                best_k = len(current_B)
                                best_B = list(current_B)
                                best_paths = new_paths
                            extended = True
                            break
    
    # Final verification and construction
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
                    paths.append(list(best_paths[(u, v)]))
                elif (v, u) in best_paths:
                    paths.append(list(reversed(best_paths[(v, u)])))
                else:
                    p = get_path(u, v)
                    if p:
                        paths.append(p)
                    else:
                        # Fallback: find any path
                        p = find_paths(u, v, 10, set())
                        if p:
                            paths.append(p[0])
                        else:
                            # Should not happen for connected graph
                            paths.append([u, v])
    
    # Ensure at least 2 branches
    if len(B) < 2:
        B = [0, 1]
        paths = [[0, 1]]
    
    # Convert to ensure correct types
    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
