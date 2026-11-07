from itertools import product

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = 12
    lcm_period = 27720  # LCM of 1..12
    
    best_m = 0
    best_L = 0
    
    for m in range(0, lcm_period):
        # Compute f(n, m)
        for L in range(1, 145):
            # Check if there exist distinct b_1..b_n in (m, m+L] with i|b_i
            # Bipartite matching
            candidates = list(range(m+1, m+L+1))
            # For each i, find which candidates are divisible by i
            # Use backtracking or Hungarian or simple DFS matching
            matched = [None] * n  # matched[i] = candidate index or -1
            used = [False] * L
            
            def dfs(i, visited):
                for j in range(L):
                    if visited[j] or not (candidates[j] % (i+1) == 0):
                        continue
                    if used[j] or dfs_match(j, visited):
                        # This is getting complex, let me use a simpler approach
                        pass
            
            # Let me use a cleaner bipartite matching
            # Left: i in 0..11 (representing i+1)
            # Right: j in 0..L-1 (representing candidate m+1+j)
            # Edge if (m+1+j) % (i+1) == 0
            
            # Use DFS-based bipartite matching
            match_to = [-1] * L  # match_to[j] = i matched to candidate j, or -1
            match_from = [-1] * n  # match_from[i] = j matched to i, or -1
            
            def try_match(i, visited):
                for j in range(L):
                    if visited[j]:
                        continue
                    if (m + 1 + j) % (i + 1) != 0:
                        continue
                    visited[j] = True
                    if match_to[j] == -1 or try_match(match_to[j], visited):
                        match_to[j] = i
                        match_from[i] = j
                        return True
                return False
            
            found = 0
            for i in range(n):
                visited = [False] * L
                if try_match(i, visited):
                    found += 1
            
            if found == n:
                best_L_candidate = L
                break
        else:
            best_L_candidate = 144  # shouldn't happen per contract
        
        if best_L_candidate > best_L:
            best_L = best_L_candidate
            best_m = m
    
    return best_m
