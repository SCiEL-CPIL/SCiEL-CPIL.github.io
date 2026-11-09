import time
import random
import math
from collections import Counter

# Precompute small primes for faster factorization
_SMALL_PRIMES = []
def _init_primes():
    is_prime = [True] * 200
    is_prime[0] = is_prime[1] = False
    for i in range(2, 200):
        if is_prime[i]:
            _SMALL_PRIMES.append(i)
            for j in range(i*i, 200, i):
                is_prime[j] = False
_init_primes()

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            primes.add(p)
            n //= p
    if n > 1:
        primes.add(n)
    return primes

# Cache for prime factorization
_prime_cache = {}

def get_primes_cached(n):
    """Return the set of prime factors of n with caching."""
    if n in _prime_cache:
        return _prime_cache[n]
    result = get_primes(n)
    _prime_cache[n] = result
    return result

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes_cached(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    def time_left():
        return deadline - time.time()

    # Initial candidates
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    candidates = []
    
    # Strategy 1: Residue classes modulo 2*p for small primes p
    # If a_i = 2*p*k + p, then a_i + a_j = 2*p*(k_i + k_j + 1)
    # This ensures 2 and p are factors.
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        mod = 2 * p
        res = p
        # Find 12 distinct numbers <= 10000 with this property
        nums = []
        # Start from smallest positive
        # x = mod*k + res
        # We want distinct x.
        # Let's just pick the first 12 valid numbers
        k = 0
        while len(nums) < 12:
            val = mod * k + res
            if val < 1:
                k += 1
                continue
            if val > 10000:
                break
            if val not in nums:
                nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)
            
    # Strategy 2: Small multiples of highly composite numbers
    # Multiples of 60, 120, etc.
    for base in [6, 12, 30, 60, 120, 210, 420, 840, 2520]:
        nums = []
        k = 1
        while len(nums) < 12:
            val = base * k
            if val > 10000:
                break
            nums.append(val)
            k += 1
        if len(nums) == 12:
            candidates.append(nums)

    # Strategy 3: Powers of 2 and scaled powers of 2
    for scale in [1, 2, 3, 5, 6, 10, 15, 30, 60, 120, 210]:
        nums = []
        for i in range(12):
            val = scale * (2 ** i)
            if val <= 10000:
                nums.append(val)
            else:
                break
        if len(nums) == 12:
            candidates.append(nums)

    # Strategy 4: Small integers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    
    # Strategy 5: Random
    for _ in range(5):
        candidates.append([random.randint(1, 10000) for _ in range(12)])

    # Evaluate all candidates
    for c in candidates:
        if time_left() < 5:
            break
        # Ensure distinctness and bounds
        c = [max(1, min(10000, x)) for x in c]
        if len(set(c)) < 12:
            # Fix duplicates by adding random distinct numbers
            existing = set(c)
            while len(c) < 12:
                v = random.randint(1, 10000)
                if v not in existing:
                    c.append(v)
                    existing.add(v)
        c = c[:12]
        
        primes = evaluate(c)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = c[:]
    
    if best_set is None:
        return list(range(1, 13))
    
    # Bitset-based clique search for P-smooth compatibility
    # Try different sets of allowed primes
    prime_sets_to_try = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    for allowed_primes in prime_sets_to_try:
        if time_left() < 10:
            break
            
        # Build compatibility graph using bitsets
        # For each number 1..10000, compute which other numbers it's compatible with
        # Two numbers a, b are compatible if all prime factors of (a+b) are in allowed_primes
        
        # Precompute: for each sum s from 2 to 20000, check if s is P-smooth
        smooth_sums = [False] * 20001
        for s in range(2, 20001):
            temp = s
            valid = True
            for p in allowed_primes:
                while temp % p == 0:
                    temp //= p
            if temp != 1:
                valid = False
            smooth_sums[s] = valid
        
        # Build adjacency bitsets for each vertex (1..10000)
        # adj[i] is a bitset where bit j is set if i and j are compatible
        # We'll use Python integers as bitsets for efficiency
        
        # For memory efficiency, we only need to store adjacencies for numbers we might use
        # Let's focus on numbers that have many compatible partners
        
        # Compute degree for each number
        degrees = [0] * 10001
        for i in range(1, 10001):
            for j in range(i + 1, 10001):
                if smooth_sums[i + j]:
                    degrees[i] += 1
                    degrees[j] += 1
        
        # Sort numbers by degree (descending) to prioritize high-degree vertices
        sorted_nums = sorted(range(1, 10001), key=lambda x: degrees[x], reverse=True)
        
        # Try to find a 12-clique starting from high-degree vertices
        found_clique = None
        for start_idx in range(min(50, len(sorted_nums))):
            if time_left() < 5:
                break
                
            start_num = sorted_nums[start_idx]
            if degrees[start_num] < 11:
                continue
                
            # Get all neighbors of start_num
            neighbors = []
            for j in range(1, 10001):
                if j != start_num and smooth_sums[start_num + j]:
                    neighbors.append(j)
            
            if len(neighbors) < 11:
                continue
            
            # Try to find 11 more vertices that form a clique with each other and with start_num
            # Use backtracking
            def find_clique(current_clique, candidates):
                if len(current_clique) == 12:
                    return current_clique[:]
                if time_left() < 2:
                    return None
                if not candidates:
                    return None
                    
                for i, num in enumerate(candidates):
                    # Check if num is compatible with all in current_clique
                    compatible = True
                    for c in current_clique:
                        if not smooth_sums[c + num]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    new_clique = current_clique + [num]
                    remaining = candidates[i+1:]
                    result = find_clique(new_clique, remaining)
                    if result is not None:
                        return result
                return None
            
            # Start with start_num, find 11 more from neighbors
            result = find_clique([start_num], neighbors)
            if result is not None:
                found_clique = result
                break
        
        if found_clique is not None:
            primes = evaluate(found_clique)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = found_clique[:]
                # If we found a clique with exactly the allowed primes, we can stop
                if len(primes) == len(allowed_primes):
                    break
    
    # Local Search to improve further
    current_set = best_set[:]
    current_primes = set(best_primes)
    current_count = best_count
    
    # Precompute a list of "good" deltas to try
    small_deltas = list(range(-20, 21))
    med_deltas = list(range(-100, 101, 5))
    
    no_improve_count = 0
    max_no_improve = 500
    
    # Precompute a pool of "smooth" candidates for replacement
    smooth_pool = []
    for i in range(1, 10001):
        if len(get_primes_cached(i)) <= 2:
            smooth_pool.append(i)
    
    while time_left() > 5:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        r = random.random()
        if r < 0.3:
            # Small perturbation
            new_val = old_val + random.choice(small_deltas)
        elif r < 0.5:
            # Medium perturbation
            new_val = old_val + random.choice(med_deltas)
        elif r < 0.8:
            # Try a smooth number from the pool
            new_val = random.choice(smooth_pool)
        else:
            # Structured move: try to align with current prime factors
            other_idx = random.randint(0, 11)
            if other_idx == idx:
                other_idx = (idx + 1) % 12
            base = current_set[other_idx]
            step = random.choice([2, 3, 5, 6, 10, 30])
            new_val = base + random.randint(-10, 10) * step
            
        if new_val < 1 or new_val > 10000:
            continue
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_set and current_set.index(new_val) != idx:
            continue
        
        # Swap
        new_set = current_set[:]
        new_set[idx] = new_val
        
        # Evaluate
        new_primes = evaluate(new_set)
        new_count = len(new_primes)
        
        if new_count < current_count:
            current_set = new_set
            current_primes = new_primes
            current_count = new_count
            no_improve_count = 0
            if current_count < best_count:
                best_count = current_count
                best_primes = current_primes
                best_set = current_set[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart
                no_improve_count = 0
                restart_set = [random.randint(1, 10000) for _ in range(12)]
                # Ensure distinct
                while len(set(restart_set)) < 12:
                    restart_set.append(random.randint(1, 10000))
                restart_set = restart_set[:12]
                restart_primes = evaluate(restart_set)
                if len(restart_primes) < current_count:
                    current_set = restart_set
                    current_primes = restart_primes
                    current_count = len(restart_primes)
                    if current_count < best_count:
                        best_count = current_count
                        best_primes = current_primes
                        best_set = current_set[:]
                else:
                    pass

    return best_set
