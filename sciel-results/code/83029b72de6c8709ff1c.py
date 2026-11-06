import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin below 90s kill

    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1  # we need LPF for n and n+1, so up to 1000001

    # 1. Sieve of Eratosthenes to identify primes
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0:2] = False
    limit = int(math.isqrt(MAX_VAL)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i:MAX_VAL+1:i] = False

    primes = np.where(is_prime)[0]

    # 2. Compute Largest Prime Factor (LPF) for all numbers
    # The largest prime that divides a number is its LPF.
    # We process primes in increasing order, so later (larger) primes overwrite.
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for p in primes:
        if p > MAX_VAL:
            break
        lpf[p::p] = p

    if time.time() > deadline:
        return LO

    # 3. Compute objective: max(LPF(n), LPF(n+1)) for each n in [LO, HI]
    lpf_n = lpf[LO:HI+1]       # LPF of n for n = LO..HI
    lpf_n1 = lpf[LO+1:HI+2]    # LPF of n+1 for n = LO..HI

    costs = np.maximum(lpf_n, lpf_n1)

    # 4. Find the global minimum
    min_idx = int(np.argmin(costs))
    best_n = LO + min_idx
    best_cost = int(costs[min_idx])

    # 5. Verify with exact trial division (as the evaluator does)
    def lpf_trial(x):
        if x < 2:
            return 0
        orig = x
        lpf_val = 1
        # trial divide by 2
        if x % 2 == 0:
            lpf_val = 2
            while x % 2 == 0:
                x //= 2
        # odd trial divisors
        d = 3
        while d * d <= x:
            if x % d == 0:
                lpf_val = d
                while x % d == 0:
                    x //= d
            d += 2
        if x > 1:
            lpf_val = max(lpf_val, x)
        return lpf_val

    # Verify the best candidate with trial division as a safety check
    n_val = best_n
    lpf1 = lpf_trial(n_val)
    lpf2 = lpf_trial(n_val + 1)
    trial_cost = max(lpf1, lpf2)
    
    # If trial division disagrees, trust trial division (shouldn't happen)
    if trial_cost != best_cost:
        # Find the true best by checking all tied candidates
        tie_mask = costs == best_cost
        tie_indices = np.where(tie_mask)[0]
        best_trial_n = best_n
        best_trial_cost = trial_cost
        for ci in tie_indices[:50]:
            nv = LO + int(ci)
            t1 = lpf_trial(nv)
            t2 = lpf_trial(nv + 1)
            tc = max(t1, t2)
            if tc < best_trial_cost:
                best_trial_cost = tc
                best_trial_n = nv
        best_n = best_trial_n

    # 6. Verify the final answer
    lpf_final1 = lpf_trial(best_n)
    lpf_final2 = lpf_trial(best_n + 1)
    final_cost = max(lpf_final1, lpf_final2)

    # Sanity check: make sure best_n is in range
    if not (LO <= best_n <= HI):
        return LO

    return int(best_n)
