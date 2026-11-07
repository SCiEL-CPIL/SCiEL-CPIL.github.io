import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 10000000

    # Compute sigma(n) for all n using a direct sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Sigma
    sieve_completed = True
    for i in range(1, N + 1):
        if time.time() > deadline - 30.0:
            sieve_completed = False
            break
        # Add i to all multiples of i
        sigma[i::i] += i

    if not sieve_completed or time.time() > deadline - 15.0:
        return [11, 6]

    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)

    # Find best pair using binary search within each group
    best_error = Fraction(1, 3)
    best_pair = (11, 6)

    # Sort groups by size descending to check larger groups first (more likely to have good pairs)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

    for s, lst in sorted_groups:
        if time.time() > deadline - 5.0:
            break
        if len(lst) < 2:
            continue
        lst.sort()
        for b in lst:
            target = 3 * b / 2.0
            pos = bisect.bisect_left(lst, target)
            for idx in [pos - 1, pos]:
                if 0 <= idx < len(lst):
                    a = lst[idx]
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    err = Fraction(num, den)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
