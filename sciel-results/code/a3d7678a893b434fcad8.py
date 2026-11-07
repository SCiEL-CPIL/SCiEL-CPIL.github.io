import time
from math import ceil

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    deadline = time.time() + 85  # safety margin
    
    # LCM of all divisors
    from math import gcd
    def lcm(a, b):
        return a // gcd(a, b) * b
    LCM = 1
    for d in divisors:
        LCM = lcm(LCM, d)
    
    best_range = float('inf')
    best_assign = None
    
    # For each window start L, find the tightest set of distinct multiples
    # For divisor d starting at L, the multiples are: m0 = ceil(L/d)*d, m0+d, m0+2d, ...
    # We need to pick one per divisor, all distinct, minimizing max-min
    
    # Search L over one full LCM period
    for L in range(0, LCM + 1):
        if time.time() > deadline:
            break
        # For each divisor, compute the first few multiples >= L
        candidates = []
        for d in divisors:
            m0 = ((L - 1) // d + 1) * d  # smallest multiple of d >= L
            # Consider a few multiples: m0, m0+d, m0+2d, ... up to L + 200
            cands = []
            m = m0
            while m <= L + 300:
                cands.append(m)
                m += d
            candidates.append(cands)
        
        # Now try to assign distinct values minimizing max-min
        # Use a simple approach: try all combinations of the first 3 candidates for each
        # Actually, with 10 divisors and ~3-5 candidates each, brute force is too big.
        # Instead: sort all candidate values, and try to find a tight window where each divisor is covered.
        
        # Alternative: for a given range R, check if we can assign. Binary search on R.
        # But let's just try: for each divisor, pick the smallest multiple >= L that hasn't been used.
        # Then the range is max - min. Try also picking slightly larger multiples for some to reduce range.
        
        # Simple heuristic: pick the smallest available multiple for each divisor, then try local adjustments.
        used = set()
        assign = []
        for i, d in enumerate(divisors):
            m0 = ((L - 1) // d + 1) * d
            m = m0
            while m in used:
                m += d
            assign.append(m)
            used.add(m)
        
        rng = max(assign) - min(assign)
        if rng < best_range:
            best_range = rng
            best_assign = assign[:]
            if best_range <= 1:  # can't do better than 1 for distinct integers
                break
    
    if best_assign is None:
        # Fallback
        best_assign = divisors[:]
    
    return list(best_assign)
