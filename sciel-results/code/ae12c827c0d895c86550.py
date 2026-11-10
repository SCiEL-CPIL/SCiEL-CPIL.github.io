import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Search for minimal k
    # Lower bound estimation:
    # Sum_{i=5}^{5+k-1} 1/i >= 1
    # For k=26, sum is approx 0.99.
    # For k=27, sum is approx 1.00.
    # So k is likely 26 or 27.
    
    for k in range(26, 35):
        if time.time() > deadline - 10:
            break
            
        result = _search(k, 4, 1, 1)
        if result:
            # Verify result just in case
            if _verify(result):
                return result
                
    # Fallback
    return _fallback_solution()

def _verify(denoms):
    if len(denoms) == 0 or len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 5 or d > 10000000 for d in denoms):
        return False
    # Check sum == 1
    # Use fractions to be exact
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in denoms:
        s += Fraction(1, d)
    return s == 1

def _search(k, last_denom, num, den):
    # Represents num/den with k terms, all > last_denom
    if k == 0:
        if num == 0:
            return []
        return None
    
    if k == 1:
        if num == 0:
            return None
        if den % num != 0:
            return None
        d = den // num
        if d > last_denom:
            return [d]
        return None
    
    # Bounds
    d_min = max(last_denom + 1, (den + num - 1) // num)
    d_max = ((k - 1) * den) // num - 1
    
    if d_min > d_max:
        return None
    
    # Heuristic: Try d values that make the new numerator divisible by something?
    # Or just iterate.
    
    for d in range(d_min, d_max + 1):
        N_new = num * d - den
        if N_new <= 0:
            continue
        
        D_new = den * d
        
        # Optional: Reduce fraction to keep numbers small
        # import math
        # g = math.gcd(N_new, D_new)
        # N_new //= g
        # D_new //= g
        
        res = _search(k - 1, d, N_new, D_new)
        if res is not None:
            return [d] + res
            
    return None

def _fallback_solution():
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
