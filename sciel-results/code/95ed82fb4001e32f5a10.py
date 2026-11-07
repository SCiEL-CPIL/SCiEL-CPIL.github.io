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
    # D[v][u] = shortest distance from v to u
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

    def find_shortest_paths(src, dst, forbidden_mask):
        """
        Find all shortest paths from src to dst avoiding vertices in forbidden_mask.
        forbidden_mask is a set of vertices.
        Returns a list of paths (lists of vertices).
        """
        if src == dst:
            return []
        if dst in forbidden_mask:
            return []
        
        # BFS to find shortest distance
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for w in adj[u]:
                if w not in dist and w not in forbidden_mask:
                    dist[w] = dist[u] + 1
                    q.append(w)
        
        if dst not in dist:
            return []
        
        target_len = dist[dst]
        paths = []
        
        def dfs(u, path):
            if len(paths) > 50: # Cap to avoid explosion
                return
            if u == dst:
                paths.append(list(path))
                return
            for w in sorted(adj[u]):
                if w in forbidden_mask:
                    continue
                if dist.get(w, -1) == dist[u] + 1:
                    path.append(w)
                    dfs(w, path)
                    path.pop()
        
        dfs(src, [src])
        return paths

    def solve_for_branches(branches, time_limit):
        """
        Attempt to find internally disjoint paths for all pairs in branches.
        Returns dict of paths or None.
        """
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Heuristic: order pairs by number of available shortest paths (ascending) to fail fast
        # We estimate "difficulty" by checking if direct edge exists or distance is small.
        # Actually, let's just try to find paths.
        
        # Sort pairs by distance (shortest first? or longest first?)
        # Shortest paths are more constrained in terms of interior vertices? 
        # Actually, longer paths have more options for interior vertices.
        # So try longer paths first? Or shortest?
        # If we use shortest paths, we use fewer interior vertices, which is good for disjointness.
        # So we prefer shortest paths.
        
        # We will use backtracking.
        # State: index of pair, set of used interior vertices.
        
        # Pre-compute all candidate paths for each pair?
        # This might be expensive. Let's do it on the fly but cache if possible.
        
        used_interior = set()
        paths = {}
        
        # Order pairs: those with fewer "options" first.
        # A pair (u,v) has options if there is a path.
        # We can estimate options by counting shortest paths?
        
        def get_options(u, v, used):
            forb = branch_set - {u, v} | used
            return find_shortest_paths(u, v, forb)

        # Sort pairs by number of options (ascending)
        # This requires calling get_options for all pairs initially, which might be slow if used is empty.
        # Let's just use a static heuristic: distance.
        # Or just iterate.
        
        # To speed up, we can sort by D[u][v] ascending.
        pairs.sort(key=lambda p: D[p[0]][p[1]])

        start_time = time.time()
        
        def backtrack(idx, used):
            if time.time() - start_time > time_limit:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forb = branch_set - {u, v} | used
            candidate_paths = find_shortest_paths(u, v, forb)
            
            if not candidate_paths:
                return False
            
            # Try each candidate path
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
    # Start with max clique
    candidates = []
    if len(best_clique) >= 2:
        candidates.append(sorted(best_clique))
        
    # Try to expand clique or find larger sets
    # High degree vertices
    high_deg = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Try all combinations of size 8, 9, 10 from top 15 high-degree vertices
    # This covers many potential "clique-like" structures
    top_k = 15
    pool = high_deg[:top_k]
    
    # We want to find the MAXIMUM size.
    # So we should try larger sizes first.
    # Max size is n=24. But unlikely.
    # Let's try sizes 10, 9, 8.
    
    # Also include the max clique and its supersets if possible?
    # No, just enumerate from the pool.
    
    # Add max clique to candidates
    # If max clique is size 7, we already have it.
    # We want to find size 8+.
    
    # Strategy:
    # Iterate target_size from 10 down to 7.
    # For each size, generate combinations from `pool`.
    # Try to solve.
    # If solved, return.
    
    # But we must also check the max clique itself (size 7) if larger ones fail.
    # And we should check if max clique can be expanded.
    
    # Let's create a list of unique candidate sets.
    # To avoid duplicates, use frozenset.
    seen = set()
    
    # 1. Max clique
    if best_clique:
        key = frozenset(best_clique)
        candidates.append(list(best_clique))
        seen.add(key)
        
    # 2. Combinations from pool
    # We prioritize larger sizes.
    # We limit the number of combinations to try per size to manage time.
    
    for size in [10, 9, 8]:
        if time.time() > deadline - 10:
            break
        # Generate combinations
        # To avoid too many, we can sample or limit based on time.
        # But for size 10 from 15, it's 3003. Manageable if solver is fast.
        # For size 9 from 15, it's 5005.
        # For size 8 from 15, it's 6435.
        
        # Let's try all of them, but with a very tight time limit per attempt.
        # If we run out of time, we stop.
        
        for combo in combinations(pool, size):
            if time.time() > deadline - 5:
                break
            key = frozenset(combo)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(list(combo))
            
        # If we found a valid one in this size, we can return immediately?
        # No, we generate all candidates first? No, that's too much memory/time.
        # Better: iterate size, then iterate combos, solve, return if success.
        
    # Wait, the loop above just adds to candidates.
    # Let's restructure:
    
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
    # We go from large to small to find the maximum.
    # But we must ensure we don't miss a size 8 if size 10 fails.
    # So we iterate size 10, 9, 8.
    # If we find a valid one, we can stop? 
    # Yes, because we are going from large to small.
    # But we must check ALL combos of size 10 before trying size 9?
    # Ideally yes. But if we run out of time, we might not check all size 10.
    # So we should interleave? Or just be fast.
    
    # Let's try size 10 first.
    for size in [10, 9, 8]:
        if time.time() > deadline - 5:
            break
        if size <= best_size:
            continue
            
        # Generate combos
        # Limit the number of combos to try if time is tight?
        # For now, try all.
        for combo in combinations(pool, size):
            if time.time() > deadline - 2:
                break
            branches = sorted(combo)
            # Quick check: are all pairs connected?
            # If D[u][v] == n for any pair, skip.
            skip = False
            for u, v in combinations(branches, 2):
                if D[u][v] == n:
                    skip = True
                    break
            if skip:
                continue
                
            paths = solve_for_branches(branches, time_limit=0.5)
            if paths:
                best_result = {'branches': branches, 'paths': [[int(x) for x in paths[(u,v)]] for u,v in combinations(branches, 2)]}
                best_size = size
                # We found one of this size. Can we stop?
                # Yes, because we are iterating sizes in descending order.
                # Any further combos of this size are same score.
                # We can break out of combo loop.
                # And break out of size loop?
                # Yes, we found a valid set of size `size`.
                # Since we want max size, and we are going down, this is the max found so far.
                # We can return immediately.
                return best_result
                
    # If we didn't return, we have best_result from max clique (size 7) or smaller.
    # But we might have missed a size 8 if we broke out of size 10/9 loops early due to time?
    # The logic above returns immediately upon finding a valid set.
    # So if we didn't return, it means no valid set was found for sizes 10, 9, 8.
    # So we return the best_result (which is size 7 or fallback).
    
    if best_result is None:
        # Fallback
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    return best_result
