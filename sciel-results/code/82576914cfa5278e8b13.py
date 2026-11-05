import time
import math

def construct():
    """
    Find two positive integers a, b with a + b = 123457 that minimize max(P(a), P(b)),
    where P(n) is the largest prime factor of n (and P(1) = 1).
    
    Strategy:
    1. Precompute largest prime factor (LPF) for every number up to target.
    2. Generate primes up to target.
    3. Incrementally build the set of K-smooth numbers as K increases over primes.
    4. At each prime threshold, check if any two smooth numbers sum to target.
    5. Return the first (smallest K) valid pair found.
    """
    target = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit

    # 1. Precompute largest prime factor using sieve
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i

    # 2. Generate primes up to target using sieve
    is_prime = [True] * (target + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(target)) + 1):
        if is_prime[i]:
            for j in range(i * i, target + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, target + 1) if is_prime[i]]

    # 3. Incrementally build smooth number set
    # smooth_nums contains all numbers <= target whose prime factors are all <= current prime
    smooth_nums = {1}

    for p in primes:
        if time.time() > deadline:
            break

        # Generate new smooth numbers involving prime p
        # For each existing smooth number x, add x*p^k for k >= 1
        old_smooth = list(smooth_nums)
        new_nums = set()
        for x in old_smooth:
            val = x
            while val * p <= target:
                val *= p
                new_nums.add(val)

        smooth_nums.update(new_nums)

        # 4. Check if any two smooth numbers sum to target
        # Iterate over smooth numbers and check if complement is also smooth
        found = False
        for a in smooth_nums:
            b = target - a
            if 1 <= b <= target and b in smooth_nums:
                # Verify: both a and b have all prime factors <= p
                # (This is guaranteed by construction, but double-check)
                if lpf[a] <= p and lpf[b] <= p:
                    return [a, b]

    # Fallback: return a valid pair
    # P(1) = 1, P(target-1) = lpf[target-1]
    return [1, target - 1]
