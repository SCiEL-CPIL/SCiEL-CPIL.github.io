"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000

    # Sieve for sigma(n) = sum of divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i

    if time.time() > deadline - 5.0:
        return [11, 6]

    # Group numbers by their sigma value
    sigma_list = sigma.tolist()
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma_list[n]].append(n)

    if time.time() > deadline - 3.0:
        return [11, 6]

    # Best known: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/6
    best_num = 1  # numerator of error (exact)
    best_den = 6  # denominator of error (exact)
    best_a, best_b = 11, 6

    for s, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 2.0:
            break
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        for i_b in range(n_len):
            b = nums_sorted[i_b]
            # ideal_a = 3b/2, so we want a close to 3b/2
            # Use integer arithmetic: |a/b - 3/2| = |2a - 3b| / (2b)
            # We want to minimize |2a - 3b| / (2b)
            # For fixed b, minimize |2a - 3b| subject to a in nums_sorted, a != b
            ideal_2a = 3 * b
            # a should be close to ideal_2a / 2 = 1.5 * b
            # bisect on a values where 2*a is close to ideal_2a
            # a close to ideal_2a/2
            target_a = ideal_2a // 2
            rem = ideal_2a % 2
            # candidates: target_a and target_a+1 (if rem==1, target_a+1 is closer)
            # Actually, let's just bisect for a near 1.5*b
            idx = bisect.bisect_left(nums_sorted, target_a)
            # Check idx and idx-1
            for j in range(max(0, idx - 1), min(n_len, idx + 1)):
                a = nums_sorted[j]
                if a == b:
                    continue
                # Error = |2a - 3b| / (2b)
                num_err = abs(2 * a - 3 * b)
                den_err = 2 * b
                # Compare num_err/den_err < best_num/best_den
                # i.e., num_err * best_den < best_num * den_err
                if num_err * best_den < best_num * den_err:
                    best_num = num_err
                    best_den = den_err
                    best_a, best_b = a, b

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
