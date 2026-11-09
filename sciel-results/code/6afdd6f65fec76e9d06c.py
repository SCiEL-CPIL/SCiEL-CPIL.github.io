import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    prime_indices = {p: i for i, p in enumerate(primes)}
    
    # We will maintain a list of (p, r)
    # Budget constraint: sum(1/p) <= 1
    # We use exact rationals for the sum check
    
    # Initial heuristic: 
    # The seed uses primes from 13 upwards. 
    # Let's try to find a better set.
    # A good starting point might be a random selection of primes that fits the budget.
    
    def random_config():
        # Try to pick a subset of primes such that sum(1/p) <= 1
        # Sort by 1/p descending (smaller primes first) to fill budget? 
        # Or just random.
        # Let's try a random subset.
        shuffled = primes[:]
        random.shuffle(shuffled)
        current_sum = Fraction(0, 1)
        config = []
        for p in shuffled:
            cost = Fraction(1, p)
            if current_sum + cost <= 1:
                current_sum += cost
                config.append([p, random.randint(0, p - 1)])
        return config, current_sum

    def calculate_coverage(config, limit=5000):
        # Returns the largest L such that 1..L are covered
        covered = [False] * (limit + 1)
        for p, r in config:
            for i in range(r, limit + 1, p):
                if i >= 1:
                    covered[i] = True
                else:
                    # If r is 0, the first positive multiple is p, 2p, etc.
                    # range(0, limit+1, p) gives 0, p, 2p...
                    # We need to mark i if i % p == r.
                    # If r==0, i% p == 0.
                    pass
        
        # Re-implement coverage marking more carefully
        covered = [False] * (limit + 1)
        for p, r in config:
            # Find the first number >= 1 such that n % p == r
            # n = k*p + r
            # We need n >= 1.
            # If r == 0, n is multiple of p. Smallest >= 1 is p (if p>=1).
            # If r > 0, smallest is r (if r>=1).
            start = r
            if start < 1:
                # r is 0. Next is p.
                start = p
            for n in range(start, limit + 1, p):
                covered[n] = True
        
        L = 0
        for i in range(1, limit + 1):
            if not covered[i]:
                L = i - 1
                break
        else:
            L = limit
        return L

    def calculate_coverage_fast(config, limit=5000):
        # Optimization: use a set or list of booleans
        covered = [False] * (limit + 1)
        for p, r in config:
            start = r if r >= 1 else p
            for n in range(start, limit + 1, p):
                covered[n] = True
        
        L = 0
        # Find first gap
        for i in range(1, limit + 1):
            if not covered[i]:
                L = i - 1
                break
        else:
            L = limit
        return L

    best_config = []
    best_score = 0
    
    # Initialize with a few random configs
    for _ in range(20):
        config, _ = random_config()
        score = calculate_coverage_fast(config, limit=2000)
        if score > best_score:
            best_score = score
            best_config = [c[:] for c in config]
            
    # Local Search
    # We will iterate until deadline
    # Moves:
    # 1. Change residue of a random prime in config
    # 2. Swap a prime in config with a prime not in config (if budget allows)
    # 3. Add a prime (if budget allows)
    # 4. Remove a prime
    
    current_config = [c[:] for c in best_config]
    current_sum = sum(Fraction(1, p) for p, r in current_config)
    current_score = best_score
    
    in_config = set(p for p, r in current_config)
    
    while time.time() < deadline:
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Change residue
            if not current_config: continue
            idx = random.randint(0, len(current_config) - 1)
            p, old_r = current_config[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r: continue
            current_config[idx][1] = new_r
            new_score = calculate_coverage_fast(current_config, limit=2000)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_config = [c[:] for c in current_config]
            else:
                # Revert
                current_config[idx][1] = old_r
                
        elif move_type == 1:
            # Swap prime
            if not current_config: continue
            idx = random.randint(0, len(current_config) - 1)
            old_p, old_r = current_config[idx]
            # Find a candidate prime not in config
            candidates = [p for p in primes if p not in in_config]
            if not candidates: continue
            new_p = random.choice(candidates)
            
            # Check budget
            old_cost = Fraction(1, old_p)
            new_cost = Fraction(1, new_p)
            if current_sum - old_cost + new_cost <= 1:
                # Perform swap
                current_config[idx] = [new_p, random.randint(0, new_p - 1)]
                in_config.remove(old_p)
                in_config.add(new_p)
                current_sum = current_sum - old_cost + new_cost
                
                new_score = calculate_coverage_fast(current_config, limit=2000)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_config = [c[:] for c in current_config]
                else:
                    # Revert
                    current_config[idx] = [old_p, old_r]
                    in_config.remove(new_p)
                    in_config.add(old_p)
                    current_sum = current_sum - new_cost + old_cost
            # else: budget exceeded, do nothing
            
        elif move_type == 2:
            # Add prime
            candidates = [p for p in primes if p not in in_config]
            if not candidates: continue
            new_p = random.choice(candidates)
            new_cost = Fraction(1, new_p)
            if current_sum + new_cost <= 1:
                new_r = random.randint(0, new_p - 1)
                current_config.append([new_p, new_r])
                in_config.add(new_p)
                current_sum += new_cost
                
                new_score = calculate_coverage_fast(current_config, limit=2000)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_config = [c[:] for c in current_config]
                else:
                    # Revert
                    current_config.pop()
                    in_config.remove(new_p)
                    current_sum -= new_cost
            # else: budget exceeded, do nothing
            
        # Optional: Remove prime (if we have too many or want to free budget)
        # But adding/swapping is usually more productive for increasing L
        # Let's add a remove move occasionally
        if random.random() < 0.1 and len(current_config) > 1:
            idx = random.randint(0, len(current_config) - 1)
            p, r = current_config[idx]
            cost = Fraction(1, p)
            current_config.pop(idx)
            in_config.remove(p)
            current_sum -= cost
            
            new_score = calculate_coverage_fast(current_config, limit=2000)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_config = [c[:] for c in current_config]
            else:
                # Revert
                current_config.insert(idx, [p, r])
                in_config.add(p)
                current_sum += cost

    return best_config
