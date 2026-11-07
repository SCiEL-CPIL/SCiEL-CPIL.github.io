import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
        
    # We need n distinct integers. So range R must be at least n-1.
    # Minimal possible range is 9.
    
    def has_valid_assignment(L, R):
        # Window is [L, L+R]
        # Candidates are L, L+1, ..., L+R
        # Size of candidate set is R+1
        if R + 1 < n:
            return None
            
        m = R + 1
        
        # adj[i] contains indices j in 0..R such that (L+j) % divisors[i] == 0
        # We use bitmasks for speed
        adj_mask = [0] * n
        for i, d in enumerate(divisors):
            mask = 0
            for j in range(m):
                if (L + j) % d == 0:
                    mask |= (1 << j)
            adj_mask[i] = mask
            
        # Bipartite matching using DFS with bitmasks
        # match[j] = i means candidate j (offset) is matched to divisor i
        match = [-1] * m
        
        def dfs(i, seen_mask):
            # Try to match divisor i
            candidates = adj_mask[i]
            # Iterate through bits set in candidates
            c = candidates
            while c:
                # Get lowest set bit
                lsb = c & -c
                j = lsb.bit_length() - 1
                c -= lsb
                
                if (seen_mask >> j) & 1:
                    continue
                seen_mask = seen_mask | (1 << j)
                
                if match[j] == -1 or dfs(match[j], seen_mask):
                    match[j] = i
                    return True
            return False
            
        for i in range(n):
            seen_mask = 0
            if not dfs(i, seen_mask):
                return None
                
        # Construct result
        result = [0] * n
        for j in range(m):
            if match[j] != -1:
                result[match[j]] = L + j
                
        return result

    # Search for minimal range R
    for R in range(n - 1, 100):
        for L in range(1, lcm + 1):
            res = has_valid_assignment(L, R)
            if res is not None:
                # Verify types and distinctness
                if len(set(res)) == n:
                    if all(isinstance(x, int) for x in res):
                        if all(res[i] % divisors[i] == 0 for i in range(n)):
                            return [int(x) for x in res]
                            
    # Fallback
    return [int(x) for x in divisors]
