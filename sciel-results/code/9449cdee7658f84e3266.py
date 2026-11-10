import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # 1. Generate first 16 primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_list = sieve(100)
    # First 15 primes
    base_primes = primes_list[:15]
    
    # Check if product fits
    prod = 1
    for p in base_primes:
        prod *= p
    if prod > 10**18:
        # Should not happen for 15 primes
        base_primes = base_primes[:14]

    k = len(base_primes)
    num_divs = 1 << k

    # Function to calculate score for a given permutation of primes
    # primes_perm: list of k primes
    def calc_score(primes_perm):
        # Generate divisors
        divs = [1]
        for p in primes_perm:
            # Use list comprehension for speed
            new_part = [d * p for d in divs]
            divs.extend(new_part)
        
        # Convert to numpy array for fast sorting and gcd
        divs_arr = np.array(divs, dtype=np.uint64)
        divs_arr.sort()
        
        # Compute gcd of adjacent pairs
        # divs_arr[:-1] and divs_arr[1:]
        gcds = np.gcd(divs_arr[:-1], divs_arr[1:])
        
        # Count how many are 1
        score = np.sum(gcds == 1)
        
        return int(score)

    # Initial state: sorted primes
    current_primes = list(base_primes)
    current_score = calc_score(current_primes)
    best_primes = list(current_primes)
    best_score = current_score
    
    # Local Search: Hill Climbing with Random Restarts
    # Since the space is small (permutations of 15 items), 
    # simple local search often gets stuck. 
    # We use a combination of systematic swaps and random perturbations.
    
    improved = True
    restarts = 0
    max_restarts = 1000 # Increased restarts due to faster evaluation
    
    while time.time() < deadline:
        if improved:
            improved = False
            # Systematic pass: try all pairs
            # To make this more effective, we can shuffle the order of pairs in each pass
            pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline: break
                # Swap i and j
                current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
                s = calc_score(current_primes)
                if s > current_score:
                    current_score = s
                    improved = True
                    # Do not revert
                else:
                    # Revert
                    current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_primes = list(current_primes)
                improved = True # Allow another pass just in case, though unlikely to improve further immediately
                
        else:
            # No improvement in systematic pass.
            # Try a random restart or perturbation
            restarts += 1
            if restarts > max_restarts:
                break
            
            # Perturb the current best state
            current_primes = list(best_primes)
            
            # Apply a few random swaps to escape local minimum
            # Number of swaps can be tuned. 3-5 is a good start.
            for _ in range(5):
                i, j = random.sample(range(k), 2)
                current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
            
            current_score = calc_score(current_primes)
            
            # Try to climb from this perturbed state
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                # Random order of pairs to diversify
                pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
                random.shuffle(pairs)
                for i, j in pairs:
                    if time.time() > deadline: break
                    current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
                    s = calc_score(current_primes)
                    if s > current_score:
                        current_score = s
                        improved_local = True
                        improved = True
                        # Accept and continue checking other pairs in this state
                        # (First Improvement strategy within the shuffle)
                    else:
                        current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
                        
            if current_score > best_score:
                best_score = current_score
                best_primes = list(current_primes)
            else:
                # Revert to best if no improvement found in this restart
                current_primes = list(best_primes)
                current_score = best_score

    # Final calculation of n
    n = 1
    for p in best_primes:
        n *= p
        
    return n
