import math
import random
import time

def construct():
    # Precompute prime factors for all numbers 1..511
    # Using a simple sieve-like approach or trial division
    prime_factors = [set() for _ in range(512)]
    for i in range(2, 512):
        if not prime_factors[i]: # i is prime
            for j in range(i, 512, i):
                prime_factors[j].add(i)
    
    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    candidates = list(range(1, 512))
    
    # Helper to check if a number is coprime to a set of prime factors
    def is_coprime_to_primes(a, prime_set):
        return not prime_factors[a].intersection(prime_set)
        
    # Helper to get the union of prime factors of a list of numbers
    def get_prime_union(nums):
        union_set = set()
        for n in nums:
            union_set.update(prime_factors[n])
        return union_set

    # 1. Greedy Initialization
    # Sort by weight descending
    sorted_candidates = sorted(candidates, key=lambda x: -weights[x])
    chosen = []
    chosen_primes = set()
    for a in sorted_candidates:
        if is_coprime_to_primes(a, chosen_primes):
            chosen.append(a)
            chosen_primes.update(prime_factors[a])
            
    current_score = sum(weights[a] for a in chosen)
    best_chosen = chosen[:]
    best_score = current_score
    best_chosen_primes = chosen_primes.copy()
    
    # 2. Simulated Annealing
    # Parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    max_time = 110.0 # Leave 10s buffer
    start_time = time.time()
    
    # Random seed for reproducibility in testing, but we want diversity
    # random.seed(42) # Optional
    
    while time.time() - start_time < max_time:
        # Generate a random move
        # Move types: 1-swap (remove 1, add 1), 2-swap (remove 2, add 2)
        # We prioritize moves that potentially increase score.
        
        # Choose move type
        if random.random() < 0.5:
            move_type = 1
        else:
            move_type = 2
            
        # Select elements to remove
        if len(chosen) < move_type:
            # Can't remove enough, just try to add if possible
            # But we need to maintain size? No, we just maximize sum.
            # If we can add without removing, that's always better.
            # Let's handle "Add" as a special case if possible.
            # Actually, if we can add a coprime number, we should.
            # The greedy init should have added all possible coprime numbers 
            # in descending order of weight, but maybe we missed some due to ordering?
            # No, greedy on sorted weights is optimal for "add if possible".
            # So we are in a local optimum for additions.
            # We must swap.
            continue
            
        remove_indices = random.sample(range(len(chosen)), move_type)
        to_remove = [chosen[i] for i in remove_indices]
        
        # Calculate primes to be freed
        freed_primes = get_prime_union(to_remove)
        
        # The remaining set
        remaining = [chosen[i] for i in range(len(chosen)) if i not in remove_indices]
        remaining_primes = get_prime_union(remaining)
        
        # We need to find `move_type` new numbers to add.
        # They must be:
        # 1. Not in `to_remove`
        # 2. Not in `remaining`
        # 3. Coprime to `remaining`
        # 4. Coprime to each other
        
        # Filter candidates
        available = []
        for a in candidates:
            if a in to_remove or a in remaining:
                continue
            if not is_coprime_to_primes(a, remaining_primes):
                continue
            available.append(a)
            
        if len(available) < move_type:
            continue
            
        # Randomly select `move_type` from available, ensuring they are coprime to each other
        # This is a small subset, so we can try random sampling
        found_adds = False
        for _ in range(100): # Try up to 100 random samples
            to_add = random.sample(available, move_type)
            # Check if to_add are coprime to each other
            valid = True
            # Union of primes of to_add must be disjoint from remaining_primes (already checked)
            # But we need to check to_add vs to_add
            # Since they are from `available`, they are coprime to `remaining`.
            # We just need to check if any two in `to_add` share a factor.
            primes_in_adds = set()
            for a in to_add:
                if primes_in_adds.intersection(prime_factors[a]):
                    valid = False
                    break
                primes_in_adds.update(prime_factors[a])
            if valid:
                found_adds = True
                break
                
        if not found_adds:
            continue
            
        # Calculate new score
        old_score_part = sum(weights[a] for a in to_remove)
        new_score_part = sum(weights[a] for a in to_add)
        delta = new_score_part - old_score_part
        
        new_score = current_score + delta
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            # Perform move
            # Update chosen list
            # Remove to_remove
            for a in to_remove:
                chosen.remove(a)
            # Add to_add
            chosen.extend(to_add)
            
            # Update chosen_primes
            # Recalculate is safer than incremental update for correctness in SA
            chosen_primes = get_prime_union(chosen)
            
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_chosen = chosen[:]
                best_chosen_primes = chosen_primes.copy()
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_chosen
