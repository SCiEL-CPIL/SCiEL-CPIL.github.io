import time

# Global cache for max base-3 value to avoid recomputing 3**n
_max_val_cache = {}

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: 
    1. Count symbol frequencies (0, 1, 2) in a single pass.
    2. Compute Chi-squared deviation from uniform distribution (n/3 each).
    3. Use a deterministic base-3 integer value of the vector as a tie-breaker.
    
    Optimization: 
    - Single Python loop to count symbols and compute the base-3 value.
    - This reduces loop overhead compared to separate counting and hashing.
    - Base-3 value calculation is faster than modular hashing.
    """
    c0 = 0
    c1 = 0
    val = 0
    
    # Single pass: Count 0s and 1s, and compute base-3 representation value
    # val represents the vector as an integer in base 3
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        val = val * 3 + v
        
    c2 = n - c0 - c1
    
    # Chi-squared deviation from uniform distribution
    ideal = n / 3.0
    if ideal > 0:
        d0 = c0 - ideal
        d1 = c1 - ideal
        d2 = c2 - ideal
        chi2 = (d0*d0 + d1*d1 + d2*d2) / ideal
    else:
        chi2 = 0.0
        
    # Primary score: Lower chi2 (more balanced) is better.
    # Scale to dominate the tie-breaker.
    balance_score = -chi2 * 1000.0
    
    # Tie-breaker: Normalize the base-3 value to [0, 1)
    max_val = _max_val_cache.get(n)
    if max_val is None:
        max_val = 3 ** n
        _max_val_cache[n] = max_val
        
    # If n is very large, 3**n might be huge, but val is also huge.
    # Division will convert to float.
    tie_breaker = val / max_val
    
    return float(balance_score + tie_breaker)
