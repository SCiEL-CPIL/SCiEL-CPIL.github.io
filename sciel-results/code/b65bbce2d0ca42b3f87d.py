import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit

    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    n = len(divisors)

    # Compute LCM of all divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    # L = lcm(6,10,14,15,21,22,26,33,35,39) = 30030

    # Key insight: the span (max - min) of the "next multiple" assignment
    # is periodic with period L. For a given residue r = M mod L, the offsets
    # o_i(r) = (d_i - r) % d_i determine the relative positions.
    # span(r) = max(o_i) - min(o_i), independent of the actual M.
    # We need all o_i to be distinct for the candidates to be distinct.
    
    # Strategy 1: Enumerate all residues r in [0, L-1] to find the minimum
    # possible span. Since L=30030, this is fast.
    
    best_span = float('inf')
    best_r = None
    best_offsets = None

    for r in range(L):
        if time.time() > deadline:
            break
        
        offsets = []
        valid = True
        for d in divisors:
            o = (d - r) % d
            offsets.append(o)
        
        # Check distinctness of offsets
        if len(set(offsets)) != n:
            continue
        
        span = max(offsets) - min(offsets)
        
        if span < best_span:
            best_span = span
            best_r = r
            best_offsets = offsets[:]
            
            if best_span <= 9:
                # Can't do better than 9 for 10 distinct integers
                break

    # Now we need to find M = q*L + r such that all x_i = M + o_i <= max_val
    # and x_i >= 1.
    # M + max_offset <= max_val => M <= max_val - max_offset
    # M + min_offset >= 1 => M >= 1 - min_offset
    
    if best_r is not None:
        max_offset = max(best_offsets)
        min_offset = min(best_offsets)
        
        # Find q such that lower <= q*L + best_r <= upper
        # lower = max(0, 1 - min_offset)
        # upper = max_val - max_offset
        
        lower_bound = max(0, 1 - min_offset)
        upper_bound = max_val - max_offset
        
        # q*L + best_r >= lower_bound => q >= ceil((lower_bound - best_r) / L)
        # q*L + best_r <= upper_bound => q <= floor((upper_bound - best_r) / L)
        
        if lower_bound - best_r <= 0:
            q_min = 0
        else:
            q_min = math.ceil((lower_bound - best_r) / L)
        
        q_max = math.floor((upper_bound - best_r) / L)
        
        if q_min <= q_max:
            # Pick the smallest q to keep values small (though any valid q works)
            q = q_min
            M = q * L + best_r
            candidates = [M + o for o in best_offsets]
            
            # Verify
            if all(1 <= c <= max_val for c in candidates) and len(set(candidates)) == n:
                for c, d in zip(candidates, divisors):
                    if c % d != 0:
                        break
                else:
                    return [int(x) for x in candidates]

    # Fallback: If the above failed (shouldn't happen), do a broader search.
    # Use the window-based approach from inspiration 1 as a backup.
    # But first, let's try a slightly different residue search if the first one failed.
    
    # Actually, the first method should always find a solution if one exists.
    # The only reason it might fail is if no distinct offsets exist for any r,
    # which is impossible for these divisors.
    
    # Just in case, implement a simple search over M
    best_result = None
    best_span2 = float('inf')
    
    # Search M in a range. Since the period is L, we can search M in [0, L + max_val]
    # but that's too large. Let's just search M in [0, 100000] or so.
    # Actually, we know the optimal span is small.
    
    # Let's try M from 0 to L + 1000
    search_limit = min(L + 1000, max_val)
    for M in range(0, search_limit + 1):
        if time.time() > deadline:
            break
        
        candidates = []
        valid = True
        for d in divisors:
            o = (d - M) % d
            x = M + o
            if x < 1 or x > max_val:
                valid = False
                break
            candidates.append(x)
        
        if not valid:
            continue
        
        if len(set(candidates)) != n:
            continue
        
        span = max(candidates) - min(candidates)
        if span < best_span2:
            best_span2 = span
            best_result = candidates[:]
            if best_span2 <= 9:
                break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Last resort fallback
    # Just return the divisors themselves, adjusted to be distinct
    result = []
    used = set()
    for d in divisors:
        val = d
        while val in used:
            val += d
        result.append(val)
        used.add(val)
    
    return [int(x) for x in result]
