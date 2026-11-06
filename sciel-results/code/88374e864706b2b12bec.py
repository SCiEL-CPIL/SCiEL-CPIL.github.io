"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def prime_factors(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1 if d == 2 else 2  # skip even numbers after 2
        if n > 1:
            factors.add(n)
        return factors
    
    def count_primes(nums):
        """Count distinct primes dividing all pairwise sums of distinct elements."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes |= prime_factors(s)
        return len(primes), primes
    
    best_count = float('inf')
    best_nums = None
    
    def try_update(nums):
        nonlocal best_count, best_nums
        c, _ = count_primes(nums)
        if c < best_count:
            best_count = c
            best_nums = nums[:]
        return c
    
    # Strategy 1: Multiples of large powers of 2
    # For base = 2^k, pick 12 distinct multiples
    for k in range(1, 14):
        if time.time() > deadline:
            break
        base = 1 << k
        max_q = 10000 // base
        if max_q < 12:
            continue
        # Try consecutive quotients
        for start_q in range(1, max_q - 11 + 1):
            if time.time() > deadline:
                break
            qs = list(range(start_q, start_q + 12))
            nums = [q * base for q in qs]
            try_update(nums)
        # Try random subsets of quotients
        for _ in range(50):
            if time.time() > deadline:
                break
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * base for q in qs]
            try_update(nums)
    
    # Strategy 2: Multiples of other highly composite / smooth numbers
    smooth_bases = [6, 10, 12, 14, 15, 18, 20, 24, 30, 36, 40, 42, 48, 60, 72, 80, 84, 90, 100, 120, 144, 168, 180, 200, 240, 300, 360, 400, 420, 480, 600, 720, 800, 840, 900, 960, 1000, 1200, 1440, 1680, 1800, 2000, 2400, 2880, 3000, 3600, 4000, 4200, 4800, 5000, 6000, 7200, 8000, 8400, 9000, 9600]
    for base in smooth_bases:
        if time.time() > deadline:
            break
        max_q = 10000 // base
        if max_q < 12:
            continue
        for start_q in range(1, min(max_q - 11, 30) + 1):
            if time.time() > deadline:
                break
            qs = list(range(start_q, start_q + 12))
            nums = [q * base for q in qs]
            try_update(nums)
        for _ in range(20):
            if time.time() > deadline:
                break
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * base for q in qs]
            try_update(nums)
    
    # Strategy 3: Local improvement from best found
    if best_nums is not None and time.time() < deadline - 10:
        nums = best_nums[:]
        c = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                # Try replacing nums[i] with nearby values
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    new_val = nums[i] + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in nums:
                        continue
                    old = nums[i]
                    nums[i] = new_val
                    c2, _ = count_primes(nums)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i] = old
    
    # Strategy 4: Random restarts with local improvement
    if time.time() < deadline - 15:
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            nums = random.sample(range(1, 10001), 12)
            c, _ = count_primes(nums)
            if c <= best_count + 2:  # only improve if close
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(12):
                        if time.time() > deadline - 5:
                            break
                        for delta in range(-10, 11):
                            if delta == 0:
                                continue
                            new_val = nums[i] + delta
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val in nums:
                                continue
                            old = nums[i]
                            nums[i] = new_val
                            c2, _ = count_primes(nums)
                            if c2 < c:
                                c = c2
                                if c < best_count:
                                    best_count = c
                                    best_nums = nums[:]
                                improved = True
                            else:
                                nums[i] = old
    
    if best_nums is None:
        return list(range(1, 13))
    
    return best_nums
# EVOLVE-BLOCK-END
