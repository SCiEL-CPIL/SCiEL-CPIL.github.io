import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
    
    # Precompute vectors for all numbers 2..40
    vecs = {}
    for n in range(2, 41):
        vecs[n] = factorize(n)
    
    def is_valid_subset(subset):
        """Check if all 2^k subset products are distinct using vector sums."""
        k = len(subset)
        if k > 22:
            return False
        
        seen = set()
        zero = (0,) * num_primes
        seen.add(zero)
        sums = [zero]
        
        for idx in range(k):
            v = vecs[subset[idx]]
            new_sums = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            sums = new_sums
        
        return True
    
    numbers = list(range(2, 41))
    best_set = primes[:]
    best_size = len(best_set)
    
    # Greedy add composites
    composites = [n for n in numbers if n not in primes]
    for c in composites:
        if time.time() > deadline:
            break
        if c in best_set:
            continue
        trial = best_set + [c]
        if is_valid_subset(trial):
            best_set = trial
            best_size = len(best_set)
    
    # Local search: try swaps and additions
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        # Try adding elements
        for c in numbers:
            if time.time() > deadline:
                break
            if c in best_set:
                continue
            trial = best_set + [c]
            if is_valid_subset(trial):
                best_set = trial
                best_size = len(best_set)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            for j in numbers:
                if j in best_set:
                    continue
                trial = best_set[:i] + best_set[i+1:] + [j]
                if is_valid_subset(trial):
                    # Only accept if it leads to same or larger set
                    best_set = trial
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        
        if not improved:
            break
    
    # Randomized search
    random.seed(12345)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random greedy
        shuffled = numbers[:]
        random.shuffle(shuffled)
        current = []
        for n in shuffled:
            if time.time() > deadline:
                break
            trial = current + [n]
            if is_valid_subset(trial):
                current = trial
        
        # Improve current
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for c in numbers:
                if time.time() > deadline:
                    break
                if c not in current:
                    trial = current + [c]
                    if is_valid_subset(trial):
                        current = trial
                        improved = True
                        break
            
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    for j in numbers:
                        if j in current:
                            continue
                        trial = current[:i] + current[i+1:] + [j]
                        if is_valid_subset(trial):
                            current = trial
                            improved = True
                            break
                    if improved or time.time() > deadline:
                        break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Final cleanup: ensure valid and return
    if not is_valid_subset(best_set):
        # Fallback to primes
        best_set = primes[:]
    
    return [int(x) for x in best_set]

