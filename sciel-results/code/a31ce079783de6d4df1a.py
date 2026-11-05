"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd, isqrt
from itertools import combinations

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _sums_use_only_primes(nums, allowed_primes):
    """Check if all pairwise sums of nums have prime factors subset of allowed_primes."""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _factorize(s)
            if not pf.issubset(allowed_primes):
                return False
    return True

def _find_numbers_for_primes(allowed_primes, count=12, max_val=10000, iterations=50000):
    """Try to find 'count' distinct positive integers <= max_val where all pairwise sums use only allowed_primes."""
    # Generate candidate numbers: numbers that are "compatible" with allowed_primes
    # A number x is a candidate if for many other candidates y, x+y uses only allowed_primes
    # Start with a greedy approach
    
    # First, find all numbers <= max_val that are "smooth" with respect to allowed_primes when added to small numbers
    # Actually, let's just do a randomized search
    
    # Generate a pool of candidate numbers
    # A good candidate: numbers that are 2^a * 3^b * 5^c * ... - small_offset, etc.
    # Simpler: just try random numbers and check compatibility
    
    pool = []
    for _ in range(iterations):
        x = random.randint(1, max_val)
        pool.append(x)
    
    # Greedy: pick numbers one by one, ensuring compatibility
    result = []
    random.shuffle(pool)
    
    for x in pool:
        if x in result:
            continue
        # Check x against all existing in result
        compatible = True
        for y in result:
            s = x + y
            pf = _factorize(s)
            if not pf.issubset(allowed_primes):
                compatible = False
                break
        if compatible:
            result.append(x)
            if len(result) == count:
                return result
    
    return None

def construct():
    """
    Find 12 distinct positive integers <= 10000 minimizing the number of distinct primes
    dividing sums of distinct-index pairs.
    """
    # Try with increasing sets of allowed primes
    # Start with smallest prime sets
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Try with just {2}
    result = _find_numbers_for_primes({2}, count=12, max_val=10000, iterations=20000)
    if result:
        return result
    
    # Try with {2, 3}
    result = _find_numbers_for_primes({2, 3}, count=12, max_val=10000, iterations=50000)
    if result:
        return result
    
    # Try with {2, 3, 5}
    result = _find_numbers_for_primes({2, 3, 5}, count=12, max_val=10000, iterations=100000)
    if result:
        return result
    
    # Try with {2, 3, 5, 7}
    result = _find_numbers_for_primes({2, 3, 5, 7}, count=12, max_val=10000, iterations=200000)
    if result:
        return result
    
    # Try with {2, 3, 5, 7, 11}
    result = _find_numbers_for_primes({2, 3, 5, 7, 11}, count=12, max_val=10000, iterations=300000)
    if result:
        return result
    
    # Fallback: use a structured approach
    # Use numbers that are all even, so 2 is always a factor
    # Then try to minimize other factors
    
    # Strategy: use numbers of form 2*k where k is chosen so sums are smooth
    # Or use a known good set
    
    # Fallback: return a set that at least has few prime factors
    # Use numbers that are all multiples of a high power of 2
    # e.g., all multiples of 1024, so sums are multiples of 1024 = 2^10
    # But we need 12 distinct multiples of 1024 <= 10000: 1024, 2048, ..., 12288 > 10000
    # So at most 9 multiples of 1024. Not enough.
    
    # Use multiples of 512: 512, 1024, ..., 512*19 = 9728. That's 19 numbers.
    # All sums are multiples of 512 = 2^9, so all sums are powers of 2 times... wait no.
    # 512*a + 512*b = 512*(a+b). So prime factors are those of 512*(a+b) = 2^9 * (a+b).
    # So prime factors = {2} union prime_factors(a+b) for a,b in {1,...,19}, a!=b.
    
    # So I need to choose 12 values from {1,...,19} such that the pairwise sums a+b have few distinct prime factors.
    # This is a smaller problem!
    
    # Let's solve the subproblem: choose 12 distinct values from 1..19 minimizing distinct primes in pairwise sums
    best_subset = None
    best_prime_count = float('inf')
    
    # Try all combinations of 12 from 19
    for subset in combinations(range(1, 20), 12):
        primes_in_sums = set()
        for a, b in combinations(subset, 2):
            s = a + b
            primes_in_sums.update(_factorize(s))
        if len(primes_in_sums) < best_prime_count:
            best_prime_count = len(primes_in_sums)
            best_subset = subset
    
    if best_subset:
        # Map back: numbers are 512 * k for k in best_subset
        result = [512 * k for k in best_subset]
        return result
    
    # Ultimate fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END
