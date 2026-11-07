import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths (distance only) for pruning
    D = [[n] * n for _ in range(n)]
    for v in range(n):
        D[v][v] = 0
        q = deque([v])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if D[v][w] > D[v][u] + 1:
                    D[v][w] = D[v][u] + 1
                    q.append(w)

    def find_paths_up_to_len(src, dst, forbidden, max_len):
        """
        Find simple paths from src to dst avoiding vertices in forbidden.
        Only considers paths of length <= max_len (in edges).
        Returns a list of paths (lists of vertices), sorted by length.
        """
        if src == dst:
            return []
        if dst in forbidden:
            return []
        
        # BFS to find shortest distance
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for w in adj[u]:
                if w not in dist and w not in forbidden:
                    dist[w] = dist[u] + 1
                    q.append(w)
        
        if dst not in dist:
            return []
        
        shortest = dist[dst]
        if shortest > max_len:
            return []
        
        paths = []
        
        def dfs(u, path, depth):
            if len(paths) > 30:
                return
            if depth > max_len:
                return
            if u == dst:
                if depth >= shortest:
                    paths.append(list(path))
                return
            for w in sorted(adj[u]):
                if w in forbidden:
                    continue
                if len(path) > 1 and w in path[:-1]:
                    continue
                if dist.get(w, n) <= depth + 1 and depth + 1 <= max_len:
                    path.append(w)
                    dfs(w, path, depth + 1)
                    path.pop()
        
        dfs(src, [src], 0)
        # Sort by length (shortest first)
        paths.sort(key=lambda p: len(p))
        return paths

    def solve_for_branches(branches, time_limit):
        """
        Attempt to find internally disjoint paths for all pairs in branches.
        Returns dict of paths or None.
        """
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Compute difficulty heuristic: number of available shortest paths
        def count_options(u, v, used):
            forb = branch_set - {u, v} | used
            paths = find_paths_up_to_len(u, v, forb, D[u][v] + 1)
            return len(paths)
        
        # Sort pairs by difficulty (fewest options first)
        # Use initial count (with no used vertices) as proxy
        pair_difficulty = []
        for u, v in pairs:
            forb = branch_set - {u, v}
            cnt = count_options(u, v, set())
            pair_difficulty.append((cnt, u, v))
        pair_difficulty.sort(key=lambda x: x[0])
        pairs = [(u, v) for _, u, v in pair_difficulty]
        
        used_interior = set()
        paths = {}
        start_time = time.time()
        
        def backtrack(idx, used):
            if time.time() - start_time > time_limit:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forb = branch_set - {u, v} | used
            
            # Try shortest paths first, then slightly longer
            candidate_paths = find_paths_up_to_len(u, v, forb, D[u][v] + 2)
            
            if not candidate_paths:
                return False
            
            for path in candidate_paths:
                interior = set(path[1:-1])
                if interior & used:
                    continue
                if interior & branch_set:
                    continue
                
                paths[(u, v)] = path
                if backtrack(idx + 1, used | interior):
                    return True
                del paths[(u, v)]
            
            return False

        if backtrack(0, set()):
            return paths
        return None

    # 1. Find Max Clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        pivot_candidates = P_set | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_set = P_set - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())
    
    # 2. Generate Candidates
    # High degree vertices
    high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Expand pool to top 20 for more combinations
    top_k = 20
    pool = high_deg[:top_k]
    
    best_result = None
    best_size = 0
    
    # Try Max Clique first
    if best_clique:
        branches = sorted(best_clique)
        paths = solve_for_branches(branches, time_limit=5.0)
        if paths:
            best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
            best_size = len(branches)
    
    # Try larger sizes from pool
    for size in [10, 9, 8]:
        if time.time() > deadline - 8:
            break
        if size <= best_size:
            continue
            
        for combo in combinations(pool, size):
            if time.time() > deadline - 3:
                break
            branches = sorted(combo)
            
            # Quick connectivity check
            skip = False
            for u, v in combinations(branches, 2):
                if D[u][v] == n:
                    skip = True
                    break
            if skip:
                continue
                
            # Quick density check: average distance should be low
            total_dist = sum(D[u][v] for u, v in combinations(branches, 2))
            num_pairs = len(list(combinations(branches, 2)))
            avg_dist = total_dist / num_pairs if num_pairs > 0 else n
            if avg_dist > 3.5:
                continue
            
            paths = solve_for_branches(branches, time_limit=0.8)
            if paths:
                best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
                best_size = size
                return best_result
                
    # If we didn't find size 8+, try size 7 combinations from pool that weren't the max clique
    if best_size < 7:
        for combo in combinations(pool, 7):
            if time.time() > deadline - 2:
                break
            branches = sorted(combo)
            skip = False
            for u, v in combinations(branches, 2):
                if D[u][v] == n:
                    skip = True
                    break
            if skip:
                continue
            total_dist = sum(D[u][v] for u, v in combinations(branches, 2))
            num_pairs = len(list(combinations(branches, 2)))
            avg_dist = total_dist / num_pairs if num_pairs > 0 else n
            if avg_dist > 3.5:
                continue
            paths = solve_for_branches(branches, time_limit=0.5)
            if paths:
                best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
                best_size = 7
                return best_result

    if best_result is None:
        # Fallback: find any edge
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    return best_result
