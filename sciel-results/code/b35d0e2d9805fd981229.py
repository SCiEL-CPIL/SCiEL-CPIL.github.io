import time

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1..12)
    
    # Precompute divisors for numbers? 
    # No, we just check i % k == 0.
    
    def can_match(m, L):
        # Check if there is a matching for interval (m, m+L]
        # Left nodes: 1..12
        # Right nodes: m+1 .. m+L
        # Edge i-b if i divides b
        
        # Build adjacency list for left nodes
        # adj[i] = list of indices in 0..L-1 corresponding to b = m+1+j such that i | b
        # Actually, we just need to know which b's are available for each i.
        
        # To speed up matching, we can represent the right side as a bitmask?
        # L can be up to 144, so 144 bits. Python ints can handle this.
        # But standard DFS matching is fine for N=12.
        
        # Let's use a simple DFS bipartite matching (Kuhn's algorithm)
        # match_to[i] = index of b assigned to i, or -1
        # match_from[j] = index of i assigned to b_j, or -1
        
        match_to = [-1] * (N + 1) # 1-indexed for i
        # We don't really need match_from if we just check if we can assign all i.
        # We iterate i from 1 to N.
        
        # For each i, we try to find an available b.
        # The set of available b's for i is { b in (m, m+L] | i divides b }
        
        # Optimization: 
        # The candidates for i are multiples of i in (m, m+L].
        # First multiple > m is m + (i - m % i) % i. If this is <= m, add i.
        # So candidates are k, k+i, k+2i... <= m+L.
        
        def dfs(i, visited):
            # Try to find a b for i
            # Generate candidates for i
            # start = m + (i - m % i) % i
            # if start <= m: start += i
            # Actually: 
            # rem = m % i
            # if rem == 0: first = m + i
            # else: first = m + (i - rem)
            
            rem = m % i
            if rem == 0:
                first = m + i
            else:
                first = m + (i - rem)
                
            # Iterate through multiples of i in (m, m+L]
            b = first
            while b <= m + L:
                idx = b - (m + 1) # 0-indexed index in the interval
                if not visited[idx]:
                    visited[idx] = True
                    # If b is not assigned to any i, or the i currently assigned to b can find another b
                    if match_to[idx] == -1 or dfs(match_to[idx], visited):
                        match_to[idx] = i
                        return True
                b += i
            return False

        # Reset match_to for each L? 
        # Wait, match_to maps b-index to i.
        # We need to reset match_to for each call to can_match?
        # Yes, because it's a fresh matching problem for this specific L.
        # But we are calling can_match repeatedly for increasing L.
        # It's better to implement the check inside the L loop or pass state.
        # Given L is small, recreating is fine.
        
        # However, the DFS above uses `match_to` which is local to can_match.
        # We need to reset it for each i? No, it's the global matching state for the current L.
        # We iterate i=1..N, and for each i, we run DFS.
        
        # Wait, the standard Kuhn's algorithm:
        # for i in 1..N:
        #   visited = [False] * L
        #   if not dfs(i, visited): return False
        # return True
        
        # Let's refactor can_match to use this.
        
        # Re-initialize match_to for the specific L
        # match_to is size L (indices 0..L-1)
        # But we are regenerating it every time can_match is called.
        # This is inefficient if we call can_match(m, L) then can_match(m, L+1).
        # We should integrate the L loop.
        
        pass # Placeholder, will rewrite logic below

    # Better approach:
    # For each m in 0..PERIOD-1:
    #   min_L = None
    #   for L in 1..200:
    #      if has_matching(m, L):
    #         min_L = L
    #         break
    #   if min_L > current_max:
    #      current_max = min_L
    #      best_m = m
    #      if current_max == 144: return best_m

    # Let's implement has_matching efficiently.
    
    def has_matching(m, L):
        # match_b_to_i: array of size L, initialized to -1
        # match_b_to_i[j] = i if b_j is matched to i
        match_b_to_i = [-1] * L
        
        def dfs(i, visited):
            # Find a b for i
            rem = m % i
            if rem == 0:
                first = m + i
            else:
                first = m + (i - rem)
                
            b = first
            while b <= m + L:
                idx = b - (m + 1)
                if not visited[idx]:
                    visited[idx] = True
                    if match_b_to_i[idx] == -1 or dfs(match_b_to_i[idx], visited):
                        match_b_to_i[idx] = i
                        return True
                b += i
            return False

        for i in range(1, N + 1):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    max_L = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    for m in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # We know L <= 144
        for L in range(1, 150):
            if has_matching(m, L):
                if L > max_L:
                    max_L = L
                    best_m = m
                    if max_L == 144:
                        return best_m
                break
                
    return best_m
