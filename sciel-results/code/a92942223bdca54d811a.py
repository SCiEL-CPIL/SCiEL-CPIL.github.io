import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 12
    K = 27720  # lcm(1..12)
    
    def can_match(r, L):
        # nums in (r, r+L]
        # We need to match 1..N to distinct nums
        # nums are r+1, ..., r+L
        # For each i, valid nums are those divisible by i
        
        # Optimization: If L < N, impossible? No, distinct b_i required.
        # But if L < N, there are fewer numbers than indices, so impossible.
        if L < N:
            return False
            
        # Build adjacency
        # adj[i] = list of indices j in 0..L-1 such that (r + 1 + j) % i == 0
        # We can use a bitmask for matched numbers? No, L can be up to ~100.
        # Simple DFS with visited array for right side.
        
        # To speed up, sort indices by number of choices?
        # Or just try 1..N
        
        adj = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            for j in range(L):
                if (r + 1 + j) % i == 0:
                    adj[i].append(j)
                    
        # If any i has no choices, return False
        for i in range(1, N+1):
            if not adj[i]:
                return False
                
        matched = [-1] * L  # matched[j] = i that matched to j, -1 if none
        # Or match[i] = j. Let's use match[i] = j.
        match = [-1] * (N+1)
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match[j] == -1 or dfs(match[j], seen):
                    match[j] = i
                    return True
            return False
            
        # Standard bipartite matching
        # Note: The standard algorithm iterates left nodes and tries to find augmenting paths.
        # We need to reset 'seen' for each left node? No, 'seen' is for the right side in a specific DFS call.
        # Actually, the standard Kuhn's algorithm is:
        # for u in left_nodes:
        #   seen = [False] * len(right_nodes)
        #   if not dfs(u, seen): return False
        #   # else update matching
        
        # Let's implement Kuhn's algorithm.
        match_r = [-1] * L  # match_r[j] = i
        
        def dfs_kuhn(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_r[j] == -1 or dfs_kuhn(match_r[j], seen):
                    match_r[j] = i
                    return True
            return False
            
        for i in range(1, N+1):
            seen = [False] * L
            if not dfs_kuhn(i, seen):
                return False
        return True

    best_val = 0
    best_r = 0
    
    for r in range(K):
        if time.time() > deadline:
            break
        # Find min L
        # L must be at least N.
        # Upper bound? 144 is safe.
        found = False
        for L in range(N, 150):
            if can_match(r, L):
                if L > best_val:
                    best_val = L
                    best_r = r
                found = True
                break
        if not found:
            # Should not happen, but just in case
            pass
            
    return int(best_r)
