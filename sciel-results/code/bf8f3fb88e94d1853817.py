import time
import math
import random

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

    # Precompute prime index mapping
    prime_to_idx = {p: i for i, p in enumerate(base_primes)}
    
    # Function to calculate score for a given permutation of primes
    # primes_perm: list of k primes
    def calc_score(primes_perm):
        # Generate divisors with their prime factor masks
        # Each divisor is represented by its index in the divisor list
        # and we track which primes divide it
        
        # Build divisors incrementally, tracking prime masks
        divs = [1]
        masks = [0]  # bitmask of which primes divide this divisor
        
        for idx, p in enumerate(primes_perm):
            bit = 1 << idx
            new_part = [d * p for d in divs]
            new_masks = [m | bit for m in masks]
            divs.extend(new_part)
            masks.extend(new_masks)
        
        # Sort divisors and their masks together
        paired = list(zip(divs, masks))
        paired.sort(key=lambda x: x[0])
        sorted_divs = [x[0] for x in paired]
        sorted_masks = [x[1] for x in paired]
        
        score = 0
        for i in range(1, num_divs):
            # Two divisors are coprime iff their prime masks have no common bits
            if (sorted_masks[i-1] & sorted_masks[i]) == 0:
                score += 1
        return score

    # Initial state: sorted primes
    current_primes = list(base_primes)
    current_score = calc_score(current_primes)
    best_primes = list(current_primes)
    best_score = current_score
    
    # Local Search: Hill Climbing with Random Restarts
    # Use bitmask-based score for speed
    
    improved = True
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline:
        if improved:
            improved = False
            # Systematic pass: try all pairs
            for i in range(k):
                if time.time() > deadline: break
                for j in range(i + 1, k):
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
            num_swaps = random.randint(3, 8)
            for _ in range(num_swaps):
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
