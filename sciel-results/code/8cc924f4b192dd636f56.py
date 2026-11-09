import time
import random
import math

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    timeout = 70  # Keep within 60s limit, leave margin for safety (evaluator kills at 90s, but we want to be safe)
    
    # Initial candidates
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    # Heuristic candidates
    candidates = []
    
    # 1. Multiples of 6 (ensures 2 and 3 are factors)
    candidates.append(list(range(6, 6*12+1, 6)))
    
    # 2. Multiples of 30 (ensures 2, 3, 5)
    candidates.append(list(range(30, 30*12+1, 30)))
    
    # 3. Powers of 2 (suggested by inspiration 1)
    candidates.append([2**i for i in range(12)])
    
    # 4. Small multiples of 3
    candidates.append(list(range(3, 3*12+1, 3)))
    
    # 5. Small multiples of 4
    candidates.append(list(range(4, 4*12+1, 4)))
    
    # 6. Small multiples of 12
    candidates.append(list(range(12, 12*12+1, 12)))
    
    # 7. Residue classes mod small numbers
    for mod in range(2, 20):
        for res in range(mod):
            nums = [x for x in range(1, 10001) if x % mod == res]
            if len(nums) >= 12:
                candidates.append(nums[:12])
    
    # 8. Random initializations
    for _ in range(5):
        candidates.append([random.randint(1, 10000) for _ in range(12)])
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() - start_time > timeout:
            break
        # Ensure distinctness and bounds
        c = [max(1, min(10000, x)) for x in c]
        if len(set(c)) < 12:
            # Fix duplicates
            while len(set(c)) < 12:
                c.append(random.randint(1, 10000))
        c = c[:12]
        
        primes = evaluate(c)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = c[:]
    
    if best_set is None:
        return list(range(1, 13))
    
    # Local Search
    # We will try to reduce the number of distinct primes
    # by changing one element at a time.
    current_set = best_set[:]
    current_primes = best_primes
    current_count = best_count
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 200
    
    # Precompute a list of "good" deltas to try
    # Small deltas are likely to preserve structure
    # Large deltas might break it but find new minima
    deltas = list(range(-100, 101)) + list(range(-1000, 1001, 100)) + list(range(-5000, 5001, 1000))
    random.shuffle(deltas)
    
    while time.time() - start_time < timeout:
        iteration += 1
        
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        # Strategy: Try values that might share prime factors with existing sums
        # Or just random.
        
        # Mix of random and delta-based
        if random.random() < 0.5:
            # Delta based
            d = random.choice(deltas)
            new_val = old_val + d
        else:
            # Random
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_set and current_set.count(new_val) == 1 and current_set.index(new_val) != idx:
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
                # Restart with a different heuristic
                no_improve_count = 0
                # Random restart
                current_set = [random.randint(1, 10000) for _ in range(12)]
                # Ensure distinct
                while len(set(current_set)) < 12:
                    current_set.append(random.randint(1, 10000))
                current_set = current_set[:12]
                current_primes = evaluate(current_set)
                current_count = len(current_primes)
                # Update global best if this restart is better
                if current_count < best_count:
                    best_count = current_count
                    best_primes = current_primes
                    best_set = current_set[:]

    # Return the best set found
    return best_set
