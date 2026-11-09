"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    target_n = 5
    target_d = 31
    
    def dfs(n, d, last_d, M, path):
        if time.time() > deadline:
            return None
        
        if n == 0:
            return path[:]
        
        if n < 0:
            return None
        
        # Next denominator k
        # 1/k <= n/d => k >= d/n
        k_min = max(last_d + 1, (d + n - 1) // n)
        
        if k_min > M:
            return None
        
        # Pruning: If the current residue n/d is greater than the sum of the largest possible unit fractions
        # we can still add (i.e., 1/k_min, 1/(k_min+1), ..., 1/M), then it's impossible.
        # However, calculating this sum every step is expensive.
        # A simpler bound: if n/d > 1/k_min, we are fine, we just need more terms.
        # If n/d is very small, we need large denominators.
        
        for k in range(k_min, M + 1):
            # New residue
            # r' = n/d - 1/k = (n*k - d) / (d*k)
            new_n = n * k - d
            new_d = d * k
            
            if new_n < 0:
                continue
            
            path.append(k)
            res = dfs(new_n, new_d, k, M, path)
            if res is not None:
                return res
            path.pop()
            
            # Additional pruning:
            # If new_n > 0, the remaining residue is new_n/new_d.
            # The smallest denominator we can use next is k+1.
            # So the largest term we can subtract is 1/(k+1).
            # If new_n/new_d > 1/(k+1), we need at least 2 more terms.
            # If new_n/new_d is very small, we might need many terms.
            # If the number of available denominators (M - k) is small, and the residue is large, prune.
            
        return None
    
    # Iterate M from 7 to 10000
    # We know d1 >= 7
    
    # Fallback solution from seed
    from fractions import Fraction
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    fallback = values
    
    for M in range(7, 10001):
        if time.time() > deadline - 5.0:
            break
        
        res = dfs(target_n, target_d, 0, M, [])
        if res is not None:
            return res
    
    return fallback
# EVOLVE-BLOCK-END
