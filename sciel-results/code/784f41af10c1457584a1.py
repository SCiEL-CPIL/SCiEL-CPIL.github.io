"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    N = 10000000
    
    # Step 1: Compute sigma(n) for all n up to N using divisor-sum sieve
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
    
    # Step 2: Group indices by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    # Step 3: For each group, find best pair (a, b) with sigma(a)=sigma(b)
    # minimizing |2a - 3b| / (2b), with a != b (since raw value 0 is invalid)
    best_err_num = None  # numerator of |2a-3b|
    best_err_den = None  # denominator of 2b
    best_a = None
    best_b = None
    
    for s, members in groups.items():
        if len(members) < 2:
            continue
        if time.time() > deadline:
            break
        members.sort()
        # For each b, find a in members closest to 1.5*b
        for b in members:
            if time.time() > deadline:
                break
            target = 1.5 * b
            # Binary search for target in members
            lo, hi = 0, len(members) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if members[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            # Check lo and lo-1
            candidates = []
            if lo < len(members):
                candidates.append(members[lo])
            if lo - 1 >= 0:
                candidates.append(members[lo - 1])
            for a in candidates:
                if a == b:
                    continue
                # Compute |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                den = 2 * b
                if best_err_num is None or diff * best_err_den < best_err_num * den:
                    best_err_num = diff
                    best_err_den = den
                    best_a = a
                    best_b = b
    
    if best_a is None:
        return [11, 6]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
