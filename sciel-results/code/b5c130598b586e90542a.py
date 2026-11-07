import time
import math
from itertools import product

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
    # L = lcm(6,10,14,15,21,22,26,33,35,39)
    # Let's compute: 
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # L = 2 * 3 * 5 * 7 * 11 * 13 = 30030

    # The key insight: since the pattern of multiples repeats every LCM(divisors) = 30030,
    # we can exploit periodicity. For any starting point M, the candidates are:
    # x_i = M + ((d_i - M % d_i) % d_i)
    # This is periodic in M with period L.
    
    # So we only need to check M in [0, L-1] (or more precisely, M mod L)
    # But we also need to ensure all x_i are distinct and <= max_val.
    
    # Strategy: For each residue r in [0, L-1], compute the "offsets" o_i = (d_i - r) % d_i
    # Then candidates are M + o_i where M = q*L + r for some q.
    # The span is max(o_i) - min(o_i), which is independent of q!
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually: x_i = M + o_i(r) where o_i(r) = (d_i - r) % d_i
    # So span = max_i (M + o_i) - min_i (M + o_i) = max_i(o_i) - min_i(o_i)
    # This is independent of M! The span depends only on r = M mod L.
    
    # So we just need to find r in [0, L-1] such that:
    # 1. All x_i = M + o_i(r) are distinct (i.e., all o_i(r) are distinct)
    # 2. x_i <= max_val for all i (this constrains M, but not the span)
    
    # The minimum possible span is at least 9 (10 distinct integers).
    
    # Let's enumerate all r in [0, L-1] and find the one with minimum span
    # where all offsets are distinct.
    
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
        
        # Check distinctness
        if len(set(offsets)) != n:
            continue
        
        span = max(offsets) - min(offsets)
        
        if span < best_span:
            best_span = span
            best_r = r
            best_offsets = offsets[:]
            
            if best_span <= 9:
                break

    # Now we need to find M = q*L + r such that all x_i = M + o_i <= max_val
    # i.e., M <= max_val - max(o_i)
    # and M >= 0 (positive integers, so M + o_i >= 1 for all i)
    
    if best_r is not None:
        max_offset = max(best_offsets)
        min_offset = min(best_offsets)
        
        # M must satisfy: M + max_offset <= max_val => M <= max_val - max_offset
        # and M + min_offset >= 1 => M >= 1 - min_offset
        # Also M = q*L + r for some integer q >= 0
        
        # Find smallest valid M
        # M = q*L + r
        # Need: 1 - min_offset <= q*L + r <= max_val - max_offset
        
        # Lower bound: q*L >= 1 - min_offset - r => q >= ceil((1 - min_offset - r) / L)
        # Upper bound: q*L <= max_val - max_offset - r => q <= floor((max_val - max_offset - r) / L)
        
        lower_q = math.ceil((1 - min_offset - best_r) / L)
        upper_q = math.floor((max_val - max_offset - best_r) / L)
        
        if lower_q < 0:
            lower_q = 0
        
        if lower_q <= upper_q:
            q = lower_q
            M = q * L + best_r
            candidates = [M + o for o in best_offsets]
            # Verify all are positive and <= max_val
            assert all(c >= 1 and c <= max_val for c in candidates)
            assert len(set(candidates)) == n
            # Verify divisibility
            for c, d in zip(candidates, divisors):
                assert c % d == 0
            return [int(x) for x in candidates]

    # Fallback: if the above didn't work, do a direct search
    # This should not happen, but just in case
    
    def next_multiple(M, d):
        return ((M + d - 1) // d) * d

    best_solution = None
    best_span2 = float('inf')
    
    # Search over M that are multiples of some divisor in a reasonable range
    for M in range(6, min(100000, max_val)):
        if time.time() > deadline:
            break
        
        candidates = []
        valid = True
        for d in divisors:
            x = next_multiple(M, d)
            if x > max_val:
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
            best_solution = candidates[:]
            
            if best_span2 <= 9:
                break

    if best_solution is None:
        # Last resort: just return the divisors themselves (they may not be distinct, but let's try)
        # Actually, let's construct a valid solution by taking multiples
        best_solution = []
        for d in divisors:
            best_solution.append(d)
        # Ensure distinctness by adjusting
        used = set()
        for i in range(n):
            val = best_solution[i]
            while val in used:
                val += divisors[i]
            best_solution[i] = val
            used.add(val)
    
    return [int(x) for x in best_solution]
