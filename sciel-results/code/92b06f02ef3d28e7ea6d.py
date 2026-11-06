"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def _factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            primes.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def _count_distinct_primes(numbers):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            all_primes |= _factorize(s)
    return len(all_primes), all_primes

def _try_multiples_of(M, max_val=10000, count=12):
    """Try using the first 'count' multiples of M."""
    if M * count > max_val:
        return None
    numbers = list(range(M, M * count + 1, M))
    if len(set(numbers)) != count or any(x > max_val for x in numbers):
        return None
    return numbers

def construct():
    candidates = []
    
    # Strategy 1: Multiples of various small numbers
    for M in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60, 120, 180, 210, 240, 360, 420, 504, 600, 840, 1008, 1260, 1680, 2520]:
        nums = _try_multiples_of(M)
        if nums:
            candidates.append(nums)
    
    # Strategy 2: Numbers of form k * 2^m for various m
    for m in range(0, 14):
        base = 2 ** m
        nums = [base * k for k in range(1, 13)]
        if all(x <= 10000 for x in nums) and len(set(nums)) == 12:
            candidates.append(nums)
    
    # Strategy 3: Use CRT to make all pair sums divisible by a set of primes
    # For primes P, we need a_i ≡ 0 (mod p) for odd p, and same parity for p=2
    # So all numbers divisible by lcm(P) for odd primes in P, and all even if 2 in P
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [3, 5, 7],
        [2, 5, 7],
        [2, 3, 7],
        [2, 3, 5, 7, 11, 13],
    ]
    for ps in prime_sets:
        M = 1
        for p in ps:
            M *= p
        nums = _try_multiples_of(M)
        if nums:
            candidates.append(nums)
    
    # Strategy 4: Powers of 2
    powers = [2 ** k for k in range(0, 14)]
    if all(x <= 10000 for x in powers[:12]):
        candidates.append(powers[:12])
    
    # Evaluate all candidates
    best_count = float('inf')
    best_numbers = None
    for nums in candidates:
        if not nums or len(set(nums)) != 12:
            continue
        if any(x < 1 or x > 10000 for x in nums):
            continue
        count, primes = _count_distinct_primes(nums)
        if count < best_count:
            best_count = count
            best_numbers = nums
    
    if best_numbers is None:
        return list(range(1, 13))
    
    return best_numbers
# EVOLVE-BLOCK-END
