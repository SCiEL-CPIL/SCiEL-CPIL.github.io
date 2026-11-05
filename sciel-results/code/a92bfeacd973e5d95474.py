import time
import random
import math

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_new_primes(k_vals):
    """
    Counts the number of distinct prime factors of (1 + k_i + k_j) 
    for i < j, excluding 2, 3, 5.
    """
    new_primes = set()
    n = len(k_vals)
    for i in range(n):
        for j in range(i + 1, n):
            s = 1 + k_vals[i] + k_vals[j]
            # We only care about primes > 5
            p_factors = get_prime_factors(s)
            for p in p_factors:
                if p > 5:
                    new_primes.add(p)
    return len(new_primes)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin within 90s limit, leaves time for evaluation overhead if any

    # We want 12 distinct k_i >= 0
    # a_i = 15 + 30 * k_i
    # Max a_i = 10000 => 15 + 30*k_i <= 10000 => 30*k_i <= 9985 => k_i <= 332
    
    # Improved Initial Solution: Use small k_i that are known to produce smooth sums
    # k_i = 0,1,2,3,4,5,6,7,8,9,10,11 gives sums 1+0+1=2, 1+0+2=3, etc.
    # These are all small and likely 5-smooth
    k_vals = list(range(12))
    current_cost = count_new_primes(k_vals)
    best_k = k_vals[:]
    best_cost = current_cost
    
    T = 5.0
    cooling_rate = 0.998  # Slower cooling for more exploration
    
    # Local Search / Simulated Annealing
    while time.time() < deadline:
        if T < 0.01:
            T = 1.0  # Restart temperature if stuck low
        
        # Pick a random index to modify
        idx = random.randint(0, 11)
        old_val = k_vals[idx]
        
        # Generate a new value
        # Bias towards small values where smooth numbers are more likely
        # Larger random walk steps for better exploration
        if random.random() < 0.15:
            # Random restart within [0, 150]
            new_val = random.randint(0, 150)
        else:
            # Random walk with larger steps
            step_size = random.choice([3, 5, 8, 12])
            new_val = max(0, min(332, old_val + random.randint(-step_size, step_size)))
        
        # Check distinctness
        # We need k_vals to be distinct
        if new_val in k_vals and new_val != old_val:
            continue

        # Apply move
        k_vals[idx] = new_val
        
        # Evaluate
        new_cost = count_new_primes(k_vals)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_k = k_vals[:]
        else:
            # Simulated Annealing
            delta = new_cost - current_cost
            # delta is integer >= 0
            if delta == 0:
                # Accept moves with same cost to explore
                if random.random() < 0.7:  # Higher acceptance for equal-cost moves
                    current_cost = new_cost
                else:
                    k_vals[idx] = old_val
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Reject move
                    k_vals[idx] = old_val
        
        T *= cooling_rate
    
    # Final Polish Phase: More aggressive local search on the best solution
    if time.time() < deadline - 2.0:
        polish_start = time.time()
        polish_deadline = min(deadline - 1.0, polish_start + 8.0)
        
        k_vals = best_k[:]
        current_cost = best_cost
        
        # Try all possible single-element moves from the best solution
        for idx in range(12):
            if time.time() > polish_deadline:
                break
            old_val = k_vals[idx]
            # Try all values in a neighborhood
            for new_val in range(max(0, old_val - 20), min(333, old_val + 21)):
                if new_val == old_val:
                    continue
                if new_val in k_vals:
                    continue
                
                k_vals[idx] = new_val
                new_cost = count_new_primes(k_vals)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_k = k_vals[:]
                    break
                else:
                    k_vals[idx] = old_val
        
        # Try pairwise swaps
        for i in range(12):
            if time.time() > polish_deadline:
                break
            for j in range(i + 1, 12):
                if time.time() > polish_deadline:
                    break
                # Try swapping values
                k_vals[i], k_vals[j] = k_vals[j], k_vals[i]
                new_cost = count_new_primes(k_vals)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_k = k_vals[:]
                else:
                    # Revert swap
                    k_vals[i], k_vals[j] = k_vals[j], k_vals[i]
    
    # Construct final array
    result = [15 + 30 * k for k in best_k]
    
    # Verify distinctness and bounds
    # k are distinct, so result are distinct.
    # Max k is 332, max result 15 + 30*332 = 9975 <= 10000.
    # Min result 15 >= 1.
    
    return result
# EVOLVE-BLOCK-END
