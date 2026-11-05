import time
from math import gcd

# Global deadline
START_TIME = time.time()
DEADLINE = 100.0  # 100s safety margin

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate primes < 512
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
        
    primes = sieve(511)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Precompute masks and weights for a in 1..511
    # Index 0 corresponds to a=1, Index 510 corresponds to a=511
    # We will use 1-based indexing for 'a' in the list, so list index i corresponds to a = i
    # But 'a' ranges 1 to 511.
    
    masks = [0] * 512  # masks[a]
    weights = [0.0] * 512 # weights[a]
    
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
        # Factorize a
        n = a
        mask = 0
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
                while n % p == 0:
                    n //= p
        if n > 1:
            # n is a prime factor
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
            else:
                # This shouldn't happen for a < 512 if primes list is complete
                pass
        masks[a] = mask

    # Sort indices by weight descending (a descending)
    # We want to consider larger 'a' first
    sorted_indices = list(range(511, 0, -1))
    
    def greedy_fill(current_set, used_mask):
        """
        Fills the set with additional coprime numbers based on sorted_indices.
        current_set: set of integers 'a'
        used_mask: bitmask of primes already used
        Returns: (new_set, new_mask, total_weight)
        """
        new_set = set(current_set)
        new_mask = used_mask
        total_weight = sum(weights[a] for a in new_set)
        
        for a in sorted_indices:
            if a in new_set:
                continue
            if masks[a] & new_mask:
                continue
            # Add a
            new_set.add(a)
            new_mask |= masks[a]
            total_weight += weights[a]
            
        return new_set, new_mask, total_weight

    # Initial Greedy Solution
    current_set, current_mask, current_weight = greedy_fill(set(), 0)
    
    best_set = current_set
    best_mask = current_mask
    best_weight = current_weight
    
    # Local Search: Remove and Refill
    improved = True
    while improved:
        if time.time() - START_TIME > DEADLINE:
            break
        improved = False
        # Iterate over a copy of the current set
        for a_to_remove in list(current_set):
            if time.time() - START_TIME > DEADLINE:
                break
            
            # Remove a_to_remove
            temp_set = current_set - {a_to_remove}
            # Recalculate mask for temp_set? 
            # It's faster to just clear the bits? No, bits might be shared.
            # We must recalculate the mask from scratch for the remaining set to be safe, 
            # or we can subtract if we know no other element has that prime.
            # Given the small size, recalculating is safer and fast enough.
            temp_mask = 0
            for a in temp_set:
                temp_mask |= masks[a]
            
            # Refill
            new_set, new_mask, new_weight = greedy_fill(temp_set, temp_mask)
            
            if new_weight > current_weight + 1e-9:
                current_set = new_set
                current_mask = new_mask
                current_weight = new_weight
                improved = True
                if current_weight > best_weight:
                    best_set = current_set
                    best_mask = current_mask
                    best_weight = current_weight
                # Restart the loop to check for further improvements from this new state
                break
        
    # If time permits, try a second pass with 2-removals?
    # Given the speed of 1-removal, it likely converges.
    # Let's add a simple 2-removal check if time remains.
    if time.time() - START_TIME < DEADLINE - 10:
        improved = True
        while improved:
            if time.time() - START_TIME > DEADLINE:
                break
            improved = False
            set_list = list(current_set)
            n_set = len(set_list)
            for i in range(n_set):
                if time.time() - START_TIME > DEADLINE:
                    break
                for j in range(i + 1, n_set):
                    a1 = set_list[i]
                    a2 = set_list[j]
                    temp_set = current_set - {a1, a2}
                    temp_mask = 0
                    for a in temp_set:
                        temp_mask |= masks[a]
                    
                    new_set, new_mask, new_weight = greedy_fill(temp_set, temp_mask)
                    
                    if new_weight > current_weight + 1e-9:
                        current_set = new_set
                        current_mask = new_mask
                        current_weight = new_weight
                        improved = True
                        if current_weight > best_weight:
                            best_set = current_set
                            best_mask = current_mask
                            best_weight = current_weight
                        break
                if improved:
                    break

    return list(best_set)

# EVOLVE-BLOCK-END
