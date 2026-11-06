"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    # Use 145s to be safe against 150s kill
    deadline = start_time + 145.0
    
    # 1. Greedy Initialization
    # Iterate from largest to smallest, pick if coprime to all existing
    chosen = []
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    current_sum = sum(1.0 / (512 - a) for a in chosen)
    best_sum = current_sum
    best_set = chosen.copy()
    
    # 2. Local Search (1-swap)
    # Try to replace a_out with a larger a_in that is coprime to the rest
    
    # Optimization: Precompute a list of all numbers 1..511
    # We will repeatedly try to improve the set.
    
    # To speed up coprimality checks, we can check against the set.
    # The set size K is ~100.
    
    improved = True
    # Limit iterations to prevent infinite loops if floating point issues arise (unlikely with integers)
    # Or just rely on time.
    max_iters = 1000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if time.time() > deadline:
            break
            
        improved = False
        iter_count += 1
        
        # Try each element as a potential removal candidate
        for i, a_out in enumerate(chosen):
            if time.time() > deadline:
                break
                
            # Temporarily remove a_out
            # We need to find best a_in in {1..511} \ (chosen \ {a_out})
            # such that gcd(a_in, b)==1 for all b in chosen \ {a_out}
            
            # Create a set of current chosen for fast lookup
            chosen_set = set(chosen)
            chosen_set.remove(a_out)
            
            # Find largest valid a_in
            best_in = -1
            # We only care if best_in > a_out because of monotonic weights
            # So we can iterate from 511 down to a_out + 1
            for a_in in range(511, a_out, -1):
                if a_in in chosen_set:
                    continue
                
                # Check if a_in is coprime to all in chosen_set
                # Quick check: if a_in is prime, it's coprime to all non-multiples.
                # But we don't have a prime sieve handy. GCD is fine.
                is_coprime = True
                for b in chosen_set:
                    if math.gcd(a_in, b) != 1:
                        is_coprime = False
                        break
                if is_coprime:
                    best_in = a_in
                    break
            
            if best_in != -1:
                # Swap
                chosen[i] = best_in
                # Update sum
                current_sum += 1.0 / (512 - best_in) - 1.0 / (512 - a_out)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = chosen.copy()
                
                improved = True
                break # Restart search with new configuration
        
    return best_set
# EVOLVE-BLOCK-END
