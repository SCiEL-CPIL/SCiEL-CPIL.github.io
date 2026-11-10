import time
import sys
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000

    # Linear sieve for sigma(n)
    # sigma[n] stores the sum of divisors of n
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    primes = []
    # exp_prime[n] stores the power of the smallest prime factor of n
    # e.g., if n = 2^3 * 5, exp_prime[n] = 8
    exp_prime = [0] * (MAX_N + 1)
    exp_prime[1] = 1
    # sigma_prime_power[n] stores sigma(p^k) where p is the smallest prime factor of n
    # e.g., if n = 2^3 * 5, sigma_prime_power[n] = 1+2+4+8 = 15
    sigma_prime_power = [0] * (MAX_N + 1)
    sigma_prime_power[1] = 1

    for i in range(2, MAX_N + 1):
        if sigma[i] == 0:
            primes.append(i)
            exp_prime[i] = i
            sigma_prime_power[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            if i * p > MAX_N:
                break
            if i % p == 0:
                # p is the smallest prime factor of i
                exp_prime[i * p] = exp_prime[i] * p
                sigma_prime_power[i * p] = sigma_prime_power[i] + exp_prime[i * p]
                sigma[i * p] = sigma[i] // sigma_prime_power[i] * sigma_prime_power[i * p]
                break
            else:
                # p is a new prime factor, and p < smallest prime factor of i
                exp_prime[i * p] = p
                sigma_prime_power[i * p] = 1 + p
                sigma[i * p] = sigma[i] * sigma_prime_power[i * p]

    if time.time() > deadline - 5.0:
        return [11, 6]

    # Group numbers by sigma value
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma[n]].append(n)

    if time.time() > deadline - 3.0:
        return [11, 6]

    best_err = float('inf')
    best_a, best_b = 11, 6
    target = 1.5

    for s, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        nums_sorted = sorted(nums)
        # For each b, find a closest to 1.5 * b
        for b in nums_sorted:
            ideal_a = 1.5 * b
            idx = bisect.bisect_left(nums_sorted, ideal_a)
            candidates = []
            if idx < len(nums_sorted):
                candidates.append(nums_sorted[idx])
            if idx > 0:
                candidates.append(nums_sorted[idx - 1])
            for a in candidates:
                if a == b:
                    continue
                err = abs(a / b - target)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b

    return [int(best_a), int(best_b)]
