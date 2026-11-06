"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    n = 12
    max_val = 10000

    # Precompute smallest prime factor (SPF) sieve up to 2*max_val
    limit = 2 * max_val
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors_set(x):
        """Return set of prime factors of x using SPF."""
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        return factors

    def score(nums):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                primes |= prime_factors_set(s)
        return len(primes)

    # Initial best: small numbers
    best = list(range(1, 13))
    best_score = score(best)
    
    # Strategy: Local search on small integers to find a set with fewer primes
    # We search for b_i in [1, 200] such that sums have few primes.
    # If found, we can scale or use directly if <= 10000.
    
    def local_search_small():
        current = list(range(1, 13))
        current_score = score(current)
        best_local = current[:]
        best_local_score = current_score
        
        no_improve = 0
        max_no_improve = 1000
        temp = 1.0
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            i = random.randint(0, n - 1)
            old_val = current[i]
            
            # Generate new value in a wider range to explore
            # Bias towards small numbers as they have smaller sums
            new_val = random.randint(1, 50)
            
            if new_val == old_val:
                continue
            if new_val in current:
                continue
                
            current[i] = new_val
            sc = score(current)
            
            # Accept if better, or with probability based on temperature (Simulated Annealing)
            if sc < current_score:
                current_score = sc
                no_improve = 0
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = current[:]
            else:
                # Probabilistic acceptance to escape local minima
                # delta = sc - current_score
                # if random() < math.exp(-delta / temp):
                #     pass
                # else:
                #     current[i] = old_val
                #     no_improve += 1
                
                # Simple hill climbing with restarts
                current[i] = old_val
                no_improve += 1
                
                if no_improve == 200:
                    # Restart with random initialization
                    current = [random.randint(1, 50) for _ in range(n)]
                    # Ensure distinct
                    while len(set(current)) < n:
                        current[random.randint(0, n-1)] = random.randint(1, 50)
                    current_score = score(current)
                    no_improve = 0
                    temp *= 0.95
                    
        return best_local, best_local_score

    # Run local search on small numbers
    if time.time() < deadline - 10:
        random.seed(42)
        small_best, small_score = local_search_small()
        if small_score < best_score:
            best_score = small_score
            # Check if small_best values are <= max_val
            if max(small_best) <= max_val:
                best = small_best[:]
            else:
                # Scale down? Or just use if valid. 
                # If any > max_val, we need to scale or find another.
                # But small_best is in [1,50], so max is 50 <= 10000.
                pass

    # If we found a set with <= 7 primes, we are likely done.
    # If we found 8, we try to improve further with larger numbers if needed.
    
    # Strategy 2: Local search on the full range [1, 10000]
    if time.time() < deadline - 5:
        random.seed(43)
        current = best[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            i = random.randint(0, n - 1)
            old_val = current[i]
            
            # Generate candidate
            # Mix of random and neighbor
            if random.random() < 0.5:
                new_val = random.randint(1, max_val)
            else:
                # Step from current
                step = random.randint(-100, 100)
                new_val = old_val + step
                if new_val < 1: new_val = 1
                if new_val > max_val: new_val = max_val
                
            if new_val in current:
                no_improve += 1
                continue
                
            current[i] = new_val
            sc = score(current)
            
            if sc < current_score:
                current_score = sc
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
            else:
                current[i] = old_val
                no_improve += 1
                
                if no_improve == 300:
                    # Perturb
                    for k in range(n):
                        current[k] = random.randint(1, max_val)
                    while len(set(current)) < n:
                        current[random.randint(0, n-1)] = random.randint(1, max_val)
                    current_score = score(current)
                    no_improve = 0

    return best
# EVOLVE-BLOCK-END
