import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors
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
        candidates = list(range(L, L + R + 1))
        
        # adj[i] contains indices j in candidates such that candidates[j] % divisors[i] == 0
        adj = [[] for _ in range(n)]
        
        for i, d in enumerate(divisors):
            # Find the first multiple of d >= L
            first = ((L + d - 1) // d) * d
            # Find the last multiple of d <= L+R
            last = ((L + R) // d) * d
            
            if first > last:
                return None
            
            # Add indices of valid multiples
            val = first
            while val <= last:
                adj[i].append(val - L)
                val += d
                
        # Bipartite matching
        # match[j] = i means candidate index j is matched to divisor i
        match = [-1] * m
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match[j] == -1 or dfs(match[j], seen):
                    match[j] = i
                    return True
            return False
            
        # Sort divisors by number of candidates to improve matching speed
        # This is a heuristic to reduce backtracking in DFS
        order = sorted(range(n), key=lambda i: len(adj[i]))
        
        for i in order:
            seen = [False] * m
            if not dfs(i, seen):
                return None
                
        # Construct result
        result = [0] * n
        for j in range(m):
            if match[j] != -1:
                result[match[j]] = candidates[j]
                
        return result

    # Search for minimal range R
    # R is the difference max - min.
    # We iterate R from n-1 upwards.
    # For each R, we iterate L from 1 to LCM.
    # Due to periodicity, if a solution exists for some L, it exists for L + k*LCM.
    # We just need to find ONE L for the minimal R.
    
    for R in range(n - 1, 200): # Upper bound 200 is safe, likely < 50
        for L in range(1, lcm + 1):
            res = has_valid_assignment(L, R)
            if res is not None:
                # Verify types and distinctness
                if len(set(res)) == n:
                    if all(isinstance(x, int) for x in res):
                        if all(res[i] % divisors[i] == 0 for i in range(n)):
                            return res
                            
    # Fallback (should not happen)
    return divisors[:]
