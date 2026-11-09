"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s
    
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for all n up to N using a sieve
    # sigma(n) = sum of all positive divisors of n
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to a simple known pair
        return [11, 6]
    
    # Step 2: Group numbers by sigma value
    # Only consider sigma values that appear at least twice
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(2, N + 1):  # skip 1 since we need a/b near 3/2, both >= 2 likely
        s = sigma[n]
        groups[s].append(n)
    
    # Check time
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Step 3: For each group with >= 2 elements, find best pair (a, b)
    # minimizing |a/b - 3/2| = |2a - 3b| / (2b)
    best_err_num = None  # numerator of |2a-3b|
    best_err_den = None  # denominator 2b
    best_a = 11
    best_b = 6
    
    for s, members in groups.items():
        if len(members) < 2:
            continue
        if time.time() > deadline - 2:
            break
        
        members.sort()
        m = len(members)
        
        # For each b in members, find a in members closest to 3b/2
        # Use binary search
        for i in range(m):
            b = members[i]
            target = 3 * b / 2.0  # we want a ≈ 3b/2
            
            # Binary search for target in members
            lo, hi = 0, m - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if members[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (if valid)
            candidates = []
            if lo < m:
                candidates.append(members[lo])
            if lo - 1 >= 0:
                candidates.append(members[lo - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # exact 3/2 is invalid
                den = 2 * b
                
                # Compare with current best
                if best_err_num is None:
                    best_err_num = num
                    best_err_den = den
                    best_a = a
                    best_b = b
                else:
                    # Compare num/den vs best_err_num/best_err_den
                    # num * best_err_den vs best_err_num * den
                    lhs = num * best_err_den
                    rhs = best_err_num * den
                    if lhs < rhs:
                        best_err_num = num
                        best_err_den = den
                        best_a = a
                        best_b = b
    
    # Ensure types are Python int
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
