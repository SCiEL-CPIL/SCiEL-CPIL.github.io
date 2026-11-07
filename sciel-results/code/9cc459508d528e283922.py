"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
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
        pred[start][start] = -1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)
    
    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    # Cache for path enumeration
    path_cache = {}
    
    def get_paths_upto(u, v, max_len, exclude_set):
        """Get all simple paths from u to v with length <= max_len, avoiding exclude_set in interior."""
        key = (u, v, max_len, frozenset(exclude_set))
        if key in path_cache:
            return path_cache[key]
        
        if u == v:
            path_cache[key] = [[u]]
            return path_cache[key]
        
        paths = []
        visited = {u}
        
        def dfs(curr, path):
            if len(path) - 1 > max_len:
                return
            if curr == v:
                # Check interior doesn't contain any excluded vertex
                interior = path[1:-1]
                if not any(x in exclude_set for x in interior):
                    paths.append(list(path))
                return
            for nxt in adj_list[curr]:
                if nxt not in visited and nxt not in exclude_set:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path)
                    path.pop()
                    visited.remove(nxt)
        
        dfs(u, [u])
        paths.sort(key=lambda p: (len(p), p))
        path_cache[key] = paths
        return paths

    start_time = time.time()
    deadline = start_time + 55.0
    
    def check_subdivision(B, time_limit=2.0):
        """Check if B has a valid clique subdivision. Returns (True, paths_dict) or (False, None)."""
        start_t = time.time()
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
        
        # For each non-adjacent pair, get candidate paths
        domains = {}
        for u, v in non_adj_pairs:
            paths = []
            for l in range(2, 7):
                p = get_paths_upto(u, v, l, B_set)
                if p:
                    paths.extend(p)
                    break
            if not paths:
                return False, None
            domains[(u, v)] = paths
        
        # MRV: sort by domain size
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        
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

    best_k = 0
    best_B = []
    best_paths = {}
    
    # Search from largest k downward
    for k in range(n, 1, -1):
        if time.time() > deadline - 3:
            break
        
        found = False
        # Systematically try combinations, but with pruning
        # Use combinations from itertools
        for B in combinations(range(n), k):
            if time.time() > deadline - 3:
                break
            
            # Quick prune: if any pair in B has no path (even allowing B vertices), skip
            B_set = set(B)
            valid = True
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v not in adj[u]:
                        # Need a path that avoids B in interior
                        # Quick check: is there any path at all?
                        if dist[u][v] == INF:
                            valid = False
                            break
                        # Check if shortest path goes through B
                        sp = get_shortest_path(u, v)
                        if sp and any(x in B_set for x in sp[1:-1]):
                            # Might still have another path, but let's be conservative
                            # Actually, let's just try the full check
                            pass
                if not valid:
                    break
            
            if not valid:
                continue
            
            ok, paths_dict = check_subdivision(list(B), time_limit=1.0)
            if ok:
                best_k = k
                best_B = list(B)
                best_paths = paths_dict
                found = True
                break  # Found one for this k, stop
        
        if found:
            break  # Found the largest k, stop searching
    
    # Fallback if nothing found (shouldn't happen)
    if best_k < 2:
        for u, v in P['edges']:
            best_B = [u, v]
            best_k = 2
            break
        if best_k < 2:
            best_B = [0, 1]
            best_k = 2
    
    # Construct output
    B = best_B
    paths_out = []
    B_set = set(B)
    
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths_out.append([int(u), int(v)])
            else:
                if (u, v) in best_paths:
                    p = best_paths[(u, v)]
                    paths_out.append([int(x) for x in p])
                elif (v, u) in best_paths:
                    p = best_paths[(v, u)]
                    paths_out.append([int(x) for x in reversed(p)])
                else:
                    # Fallback: shortest path
                    sp = get_shortest_path(u, v)
                    if sp:
                        paths_out.append([int(x) for x in sp])
                    else:
                        paths_out.append([int(u), int(v)])
    
    return {'branches': [int(b) for b in B], 'paths': paths_out}

# EVOLVE-BLOCK-END
