import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, use 50s for safety
    
    n = 12
    max_L = 144  # Prompt says raw value is at most 144
    lcm_1_12 = 27720
    
    def can_match(m, L):
        # Left nodes: 0..11 (indices for i=1..12)
        # Right nodes: 0..L-1 (indices for numbers m+1..m+L)
        # adj[i] contains list of right node indices j such that (m+1+j) % (i+1) == 0
        
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            divisor = i + 1
            # Find multiples of divisor in (m, m+L]
            # First multiple > m:
            # k = m // divisor + 1
            # val = k * divisor
            # while val <= m + L:
            #   j = val - (m + 1)
            #   adj[i].append(j)
            #   val += divisor
            
            first_k = m // divisor + 1
            val = first_k * divisor
            limit = m + L
            while val <= limit:
                j = val - (m + 1)
                if 0 <= j < L:
                    adj[i].append(j)
                val += divisor
        
        # Bipartite matching
        # matchR[j] = i if right node j is matched to left node i, else -1
        matchR = [-1] * L
        
        def dfs(u, visited):
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    if matchR[v] == -1 or dfs(matchR[v], visited):
                        matchR[v] = u
                        return True
            return False
        
        result = 0
        for u in range(n):
            visited = [False] * L
            if dfs(u, visited):
                result += 1
            else:
                return False
        return result == n

    best_m = 0
    best_L = 0
    
    # Iterate over all residues modulo 27720
    # We can sample if time is tight, but 27720 is small.
    # Binary search for min L for each m.
    
    for m in range(lcm_1_12):
        if time.time() > deadline - 5.0:
            break
            
        # Binary search for smallest L in [1, max_L]
        lo = 1
        hi = max_L
        min_L_for_m = max_L + 1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L_for_m <= max_L:
            if min_L_for_m > best_L:
                best_L = min_L_for_m
                best_m = m
                # If we found the theoretical max, we can stop early?
                # The prompt says max is 144. If we find 144, it's optimal.
                if best_L == max_L:
                    # We can try to find another one or just return.
                    # Returning immediately is safe if 144 is the max possible.
                    # However, we should ensure 144 is indeed the max.
                    # The prompt states "which is at most 144".
                    # So 144 is the ceiling.
                    break
    
    return int(best_m)
