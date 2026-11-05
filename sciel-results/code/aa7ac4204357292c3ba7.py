import time
import random
import math

def _factorize(n):
    """Return a set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct prime factors of all pairwise sums."""
    prime_set = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            prime_set.update(_factorize(s))
    return len(prime_set), prime_set

def _generate_smooth_candidates(limit=10000):
    """Generate 2-3-smooth numbers up to limit."""
    smooths = []
    i2 = 0
    i3 = 0
    val = 1
    while val <= limit:
        smooths.append(val)
        # Next power of 2
        if val * 2 <= limit and (i2 == 0 or val * 2 > val * 3 or val % 3 != 0):
            # This logic is tricky, simpler to just iterate
            pass
        val = val * 2
        i2 += 1
    
    # Simpler generation
    smooths = set()
    for i in range(15): # 2^14 > 10000
        p2 = 2 ** i
        for j in range(15): # 3^14 > 10000
            p3 = 3 ** j
            prod = p2 * p3
            if prod > limit:
                break
            smooths.add(prod)
    return sorted(list(smooths))

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Candidates: 2-3 smooth numbers are good starting points
    candidates = _generate_smooth_candidates(10000)
    
    # Initial set: Random sample from candidates
    # Ensure distinctness
    current_set = random.sample(candidates, 12)
    current_set.sort()
    
    current_count, current_primes = _count_distinct_primes(current_set)
    best_set = current_set.copy()
    best_count = current_count
    best_primes = current_primes.copy()
    
    # Local Search
    # We want to minimize best_count.
    # If counts are equal, we might prefer smaller primes or more "smooth" sums, 
    # but count is the primary metric.
    
    while time.time() < deadline:
        # Pick a random index to replace
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try to replace with a random candidate
        # Bias towards smaller numbers? Or random?
        new_val = random.choice(candidates)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in current_set:
            continue
            
        # Create new set
        new_set = current_set.copy()
        new_set[idx] = new_val
        new_set.sort()
        
        # Evaluate
        new_count, new_primes = _count_distinct_primes(new_set)
        
        # Accept if better, or equal (to explore)
        if new_count < best_count:
            best_count = new_count
            best_set = new_set
            best_primes = new_primes
            current_set = new_set
            current_count = new_count
            current_primes = new_primes
        elif new_count == best_count:
            # Accept with some probability to escape local minima?
            # Or accept if it allows future improvements?
            # Simple: accept if new_primes is a subset of best_primes?
            if new_primes.issubset(best_primes):
                current_set = new_set
                current_count = new_count
                current_primes = new_primes
                best_set = new_set
                best_primes = new_primes
            elif random.random() < 0.1:
                current_set = new_set
                current_count = new_count
                current_primes = new_primes
        else:
            # Worse, reject
            pass

    return best_set
