# EVOLVE-BLOCK-START
import time
from math import gcd

MAX_N = 10_000_000

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # 1) Compute sigma(n) for all n in [1, MAX_N] via divisor-sum sieve
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d

    # 2) Group numbers by their sigma value
    groups = {}
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]

    # 3) For each group with >=2 elements, find pair with ratio closest to 3/2
    #    Use two-pointer on sorted list: target ratio r = 3/2
    #    We want min |a/b - 3/2| over a,b in group, a != b
    #    Equivalently min |2a - 3b| / (2b)
    best_err_num = None  # numerator of |a/b - 3/2| = |2a - 3b|
    best_err_den = None  # denominator = 2*b
    best_pair = None

    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        if time.time() > DEADLINE - 2.0:
            break
        lst.sort()  # ascending
        m = len(lst)
        # Two-pointer: i from 0..m-1, j from 0..m-1, we want lst[i]/lst[j] ~ 3/2
        # i.e. 2*lst[i] ~ 3*lst[j]
        # For each j, find i such that lst[i] ~ 1.5 * lst[j]
        # Use binary search or two-pointer
        # Two-pointer: i increases as j increases (since ratio target > 1)
        i = 0
        for j in range(m):
            # We want lst[i] / lst[j] close to 3/2
            # lst[i] should be ~ 3/2 * lst[j]
            # Since lst is sorted, as j increases, target lst[i] increases
            while i < m and 2 * lst[i] < 3 * lst[j]:
                i += 1
            # Check candidates around i
            for ii in (i - 1, i):
                if 0 <= ii < m and ii != j:
                    a = lst[ii]
                    b = lst[j]
                    # error = |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue  # exact 0 is invalid
                    if best_err_num is None or (num * best_err_den < best_err_num * den):
                        best_err_num = num
                        best_err_den = den
                        best_pair = (a, b)

    if best_pair is None:
        # Fallback: return something valid (shouldn't happen)
        return [11, 6]

    a, b = best_pair
    return [int(a), int(b)]
# EVOLVE-BLOCK-END
