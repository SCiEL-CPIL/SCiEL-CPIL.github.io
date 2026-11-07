import time
import math
from functools import reduce

def construct():
    """
    Finds 10 distinct positive integers <= 1000000, each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
    such that max(reps) - min(reps) is minimized.
    
    Strategy:
    1. The pattern of multiples repeats every LCM(divisors).
    2. We search for the minimum span in the first period [1, LCM].
    3. For each candidate minimum L (which must be a multiple of some divisor d_k),
       we fix x_k = L and greedily assign the smallest available multiples to the rest.
    4. To minimize the maximum value (and thus the span), we resolve collisions by
       assigning the shared value to the divisor with the larger step size. This is
       achieved by processing divisors in descending order of their value.
    """
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    # Compute LCM of all divisors
    lcm_val = reduce(lambda a, b: a * b // math.gcd(a, b), divisors)
    
    # Search range for L: [1, LCM]
    # L must be a multiple of at least one divisor.
    max_L_search = min(lcm_val, max_val)
    
    # Generate candidate Ls
    candidate_Ls = set()
    for d in divisors:
        candidate_Ls.update(range(d, max_L_search + 1, d))
    
    sorted_Ls = sorted(candidate_Ls)
    
    # Pre-sort indices by divisor value in descending order for optimal collision resolution
    # Larger divisors should take the "colliding" slot to minimize the bump for smaller divisors.
    indices_desc = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    # Also keep ascending just in case, though descending is theoretically better.
    indices_asc = sorted(range(n), key=lambda i: divisors[i])
    
    best_span = float('inf')
    best_result = None

    for L in sorted_Ls:
        if time.time() > deadline:
            break
        
        # Try each divisor k as the one that takes the value L
        for k, d_k in enumerate(divisors):
            if L % d_k != 0:
                continue
            
            # We will try both orders, but prioritize descending.
            # In practice, descending is optimal. We can just use descending to save time,
            # but trying both is cheap and safe.
            for order in [indices_desc, indices_asc]:
                used = {L}
                reps = [0] * n
                reps[k] = L
                valid = True
                max_rep = L
                
                for i in order:
                    if i == k:
                        continue
                    d = divisors[i]
                    # Find smallest multiple of d >= L
                    m = L
                    rem = m % d
                    if rem != 0:
                        m += d - rem
                    
                    # Resolve collision: skip if m is already used
                    while m in used:
                        m += d
                    
                    if m > max_val:
                        valid = False
                        break
                    
                    used.add(m)
                    reps[i] = m
                    if m > max_rep:
                        max_rep = m
                
                if valid:
                    span = max_rep - L
                    if span < best_span:
                        best_span = span
                        best_result = reps[:]
                        # Early exit if we find a very good solution
                        if best_span <= 10:
                            return [int(x) for x in best_result]

    if best_result is None:
        # Fallback: should not happen, but just in case
        best_result = divisors[:]
        
    return [int(x) for x in best_result]
