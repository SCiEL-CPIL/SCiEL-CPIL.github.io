import time
import random
from math import isqrt

def _factorize(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_primes_in_sums(nums):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            all_primes |= _factorize(s)
    return len(all_primes)

def _count_primes_in_coefs(coefs):
    """Count distinct prime factors of sums of co-factors (for multiples-of-6 approach)."""
    all_primes = set()
    n = len(coefs)
    for i in range(n):
        for j in range(i + 1, n):
            s = coefs[i] + coefs[j]
            all_primes |= _factorize(s)
    return len(all_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_count = float('inf')
    best_nums = None
    
    def update_best(nums):
        nonlocal best_count, best_nums
        c = _count_primes_in_sums(nums)
        if c < best_count:
            best_count = c
            best_nums = list(nums)
            return True
        return False
    
    # Strategy: use multiples of 6, optimize co-factors
    # nums[i] = 6 * coefs[i], so sums are 6*(coefs[i]+coefs[j])
    # Prime factors of sum = {2, 3} union prime_factors(coefs[i]+coefs[j])
    # Total distinct primes = 2 + distinct_primes_in_coefs_sums (minus overlap with {2,3})
    
    def local_search_coefs(coefs, time_limit):
        """Local search on co-factors to minimize prime count in their sums."""
        coefs = list(coefs)
        current_count = _count_primes_in_coefs(coefs)
        best_local = current_count
        best_local_coefs = list(coefs)
        
        while time.time() < time_limit:
            improved = False
            
            # Try all pairwise swaps
            for i in range(12):
                for j in range(i + 1, 12):
                    coefs[i], coefs[j] = coefs[j], coefs[i]
                    new_count = _count_primes_in_coefs(coefs)
                    if new_count < best_local:
                        best_local = new_count
                        best_local_coefs = list(coefs)
                        improved = True
                    else:
                        coefs[i], coefs[j] = coefs[j], coefs[i]
            
            # Try replacing one element with nearby values
            for i in range(12):
                old_val = coefs[i]
                # Try a range of replacements
                candidates = set()
                for delta in range(-15, 16):
                    v = old_val + delta
                    if 1 <= v <= 1666:  # 6*1666 = 9996 <= 10000
                        candidates.add(v)
                # Also try some random values
                for _ in range(5):
                    v = random.randint(1, 1666)
                    candidates.add(v)
                
                for new_val in candidates:
                    if new_val in coefs:
                        continue
                    coefs[i] = new_val
                    new_count = _count_primes_in_coefs(coefs)
                    if new_count < best_local:
                        best_local = new_count
                        best_local_coefs = list(coefs)
                        improved = True
                    else:
                        coefs[i] = old_val
            
            if not improved:
                # Perturb and continue
                idx = random.randint(0, 11)
                coefs[idx] = random.randint(1, 1666)
                # Make sure it's distinct
                while coefs[idx] in coefs[:idx] + coefs[idx+1:]:
                    coefs[idx] = random.randint(1, 1666)
        
        # Update global best
        nums = [6 * c for c in best_local_coefs]
        if len(set(nums)) == 12 and all(1 <= x <= 10000 for x in nums):
            update_best(nums)
        
        return best_local_coefs, best_local
    
    # Multiple starting points
    start_points = [
        list(range(1, 13)),
        list(range(2, 14)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
    ]
    
    # Add random starts
    for _ in range(8):
        random_start = sorted(random.sample(range(1, 300), 12))
        start_points.append(random_start)
    
    for sp in start_points:
        if time.time() > deadline:
            break
        if len(set(sp)) != 12 or any(x < 1 for x in sp) or any(x > 1666 for x in sp):
            continue
        local_search_coefs(sp, min(time.time() + 4, deadline))
    
    # If we have a good result, do more refinement
    if best_nums is not None and best_count <= 9:
        best_coefs = [x // 6 for x in best_nums]
        local_search_coefs(best_coefs, deadline)
    
    if best_nums is None:
        return [6 * i for i in range(1, 13)]
    
    assert len(set(best_nums)) == 12
    assert all(1 <= x <= 10000 for x in best_nums)
    
    return best_nums
