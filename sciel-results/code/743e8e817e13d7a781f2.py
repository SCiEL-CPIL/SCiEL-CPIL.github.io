import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors to exploit periodicity
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
        
    def has_valid_assignment(L, R):
        # Quick pruning: check if each divisor has at least one multiple in [L, L+R]
        for d in divisors:
            if (L + R) // d == (L - 1) // d:
                return None
                
        # Build adjacency list
        # adj[i] contains offsets k (0..R) such that (L+k) % divisors[i] == 0
        adj = [[] for _ in range(n)]
        for k in range(R + 1):
            val = L + k
            for i in range(n):
                if val % divisors[i] == 0:
                    adj[i].append(k)
                    
        # Bipartite matching
        # match[k] = i means offset k is matched to divisor i
        match = [-1] * (R + 1)
        
        def dfs(i, seen):
            for k in adj[i]:
                if seen[k]:
                    continue
                seen[k] = True
                if match[k] == -1 or dfs(match[k], seen):
                    match[k] = i
                    return True
            return False
            
        for i in range(n):
            seen = [False] * (R + 1)
            if not dfs(i, seen):
                return None
                
        # Construct result
        result = [0] * n
        for k in range(R + 1):
            if match[k] != -1:
                result[match[k]] = L + k
                
        return result

    # Search for minimal range R
    for R in range(n - 1, 100):
        for L in range(1, lcm + 1):
            res = has_valid_assignment(L, R)
            if res is not None:
                # Verify distinctness and divisibility
                if len(set(res)) == n:
                    if all(res[i] % divisors[i] == 0 for i in range(n)):
                        return res
                            
    # Fallback (should not happen)
    return divisors[:]
