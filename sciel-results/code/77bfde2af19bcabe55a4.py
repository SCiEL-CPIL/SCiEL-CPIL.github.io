import random
import time
from math import isqrt

def _init_prime_factors(max_n):
    """Pre-compute prime factors for all numbers up to max_n."""
    prime_factors = [set() for _ in range(max_n + 1)]
    for i in range(2, max_n + 1):
        if not prime_factors[i]:
            # i is prime
            for j in range(i, max_n + 1, i):
                prime_factors[j].add(i)
    return prime_factors

def construct():
    """
    Find 12 distinct positive integers <= 10000 minimizing the number of distinct primes
    dividing sums of distinct-index pairs.
    """
    max_val = 10000
    count = 12
    max_sum = 2 * max_val
    
    # Pre-compute prime factors for all possible sums
    pf_cache = _init_prime_factors(max_sum)
    
    def evaluate(nums):
        """Return the set of all distinct prime factors of pairwise sums."""
        all_primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                all_primes.update(pf_cache[s])
        return all_primes

    # Greedy Initialization
    def greedy_init():
        nums = []
        # Start with a random small number to anchor
        # Or start with 2, 4, 6...?
        # Let's try starting with a random number
        first = random.randint(1, max_val)
        nums.append(first)
        
        for _ in range(count - 1):
            best_candidate = None
            best_prime_count = float('inf')
            # Try a sample of candidates to speed up
            candidates = random.sample(range(1, max_val + 1), 500)
            for c in candidates:
                if c in nums:
                    continue
                # Check if adding c increases prime count
                # We need to check sums c + x for all x in nums
                new_primes = set()
                for x in nums:
                    new_primes.update(pf_cache[c + x])
                
                # Current primes
                cur_primes = evaluate(nums)
                # Actually, we should maintain current primes incrementally in greedy
                # But for simplicity, let's just compute the union
                # This is O(N * 500 * Factorize). Factorize is O(1) lookup.
                # So 12 * 500 = 6000 lookups per step. 12 steps = 72000 lookups. Very fast.
                
                # Wait, `evaluate(nums)` inside the loop is O(N^2) = 144 lookups.
                # So total per candidate: 144 (current) + 12 (new) lookups.
                # Total per step: 500 * 156 = 78000 lookups.
                # Total for 12 steps: ~1M lookups. This is fine.
                
                # Optimization: Maintain current_primes set
                # Let's refactor greedy_init to be efficient
                pass
                
            # Let's just do a simpler greedy:
            # Maintain current_primes
            # For each candidate, calculate new_primes = current_primes union sum_factors(c, x) for all x in nums
            # If len(new_primes) < best_prime_count, update
            
            # Re-implementing greedy_init properly
            break # Dummy break to replace with proper logic below
            
        return nums

    # Proper Greedy Init
    def greedy_init_proper():
        nums = []
        current_primes = set()
        
        # First number: random
        first = random.randint(1, max_val)
        nums.append(first)
        
        for _ in range(count - 1):
            best_candidate = None
            best_new_primes = None
            best_count = float('inf')
            
            # Sample candidates
            # To encourage diversity, maybe sample from different ranges?
            # Random sample is fine.
            candidates = random.sample(range(1, max_val + 1), 1000)
            
            for c in candidates:
                if c in nums:
                    continue
                
                # Calculate primes introduced by c
                added_primes = set()
                for x in nums:
                    added_primes.update(pf_cache[c + x])
                
                new_primes = current_primes.union(added_primes)
                cnt = len(new_primes)
                
                if cnt < best_count:
                    best_count = cnt
                    best_candidate = c
                    best_new_primes = new_primes
            
            if best_candidate is not None:
                nums.append(best_candidate)
                current_primes = best_new_primes
            else:
                # Fallback: pick any random valid number
                while True:
                    c = random.randint(1, max_val)
                    if c not in nums:
                        nums.append(c)
                        # Update primes
                        for x in nums[:-1]:
                            current_primes.update(pf_cache[c + x])
                        break
                        
        return nums, current_primes

    # Run multiple greedy inits to find a good start
    best_init_nums = None
    best_init_count = float('inf')
    
    for _ in range(20):
        nums, primes = greedy_init_proper()
        cnt = len(primes)
        if cnt < best_init_count:
            best_init_count = cnt
            best_init_nums = nums
            best_init_primes = primes
            
    current_nums = best_init_nums
    current_primes = best_init_primes
    best_nums = list(current_nums)
    best_prime_count = best_init_count
    
    # Local Search
    T = 10.0
    cooling = 0.9999
    start_time = time.time()
    max_time = 55.0
    
    # To speed up local search, we can pre-compute the prime factors of sums for the current set?
    # No, just re-evaluate.
    
    while time.time() - start_time < max_time:
        idx = random.randint(0, count - 1)
        old_val = current_nums[idx]
        
        # Generate new value
        # Try to stay in the same "region" or jump
        if random.random() < 0.1:
            new_val = random.randint(1, max_val)
        else:
            # Local move
            offset = random.randint(-100, 100)
            new_val = old_val + offset
            if new_val < 1 or new_val > max_val:
                new_val = random.randint(1, max_val)
                
        if new_val == old_val or new_val in current_nums:
            continue
            
        new_nums = list(current_nums)
        new_nums[idx] = new_val
        
        new_primes = evaluate(new_nums)
        new_count = len(new_primes)
        
        if new_count < best_prime_count:
            best_prime_count = new_count
            best_nums = new_nums
            current_nums = new_nums
            current_primes = new_primes
            T = 10.0
        elif new_count == best_prime_count:
            if random.random() < 0.5:
                current_nums = new_nums
                current_primes = new_primes
        else:
            delta = new_count - best_prime_count
            if random.random() < (2.71828 ** (-delta / T)):
                current_nums = new_nums
                current_primes = new_primes
                
        T *= cooling
        
    return best_nums
