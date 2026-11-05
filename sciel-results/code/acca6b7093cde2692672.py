import math
import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Precompute weights and prime factors
    N = 512
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Factorization for fast gcd check
    # We just use math.gcd
    
    def score(chosen):
        return sum(weights[a] for a in chosen)
    
    def is_coprime_set(chosen, a):
        for b in chosen:
            if math.gcd(a, b) != 1:
                return False
        return True

    best_set = []
    best_score = 0.0
    
    # Strategy 1: Greedy with specific exclusions
    exclusions_to_try = [set(), {510}, {511}, {510, 511}, {510, 509}, {511, 509}]
    
    for excl in exclusions_to_try:
        if time.time() > deadline:
            break
        chosen = []
        for a in range(511, 0, -1):
            if a in excl:
                continue
            if is_coprime_set(chosen, a):
                chosen.append(a)
        
        s = score(chosen)
        if s > best_score:
            best_score = s
            best_set = chosen[:]
            
    # Strategy 2: Local Search / Hill Climbing
    # Start from best_set
    current_set = best_set[:]
    current_score = best_score
    no_improvement_count = 0
    
    # Helper to get unused numbers
    def get_unused(chosen):
        chosen_set = set(chosen)
        return [a for a in range(511, 0, -1) if a not in chosen_set]
    
    while time.time() < deadline - 5.0:
        improved = False
        unused = get_unused(current_set)
        
        # Try adding any unused number
        for a in unused:
            if is_coprime_set(current_set, a):
                new_set = current_set + [a]
                new_score = current_score + weights[a]
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break # Restart loop to update unused
            
        if improved:
            no_improvement_count = 0
            continue
            
        # Try swapping: remove one, add one or more
        # Simple swap: remove a, try to add b
        # We prioritize removing low weight elements
        current_set_sorted = sorted(current_set, key=lambda x: weights[x])
        
        for a_to_remove in current_set_sorted:
            if time.time() > deadline - 5.0:
                break
            temp_set = [x for x in current_set if x != a_to_remove]
            temp_score = current_score - weights[a_to_remove]
            
            # Try to add as many as possible from unused
            added_any = False
            for b in unused:
                if is_coprime_set(temp_set, b):
                    # Check if it's worth it
                    # Just add it if it increases score
                    if temp_score + weights[b] > temp_score:
                        temp_set.append(b)
                        temp_score += weights[b]
                        added_any = True
                        # Don't break, try to add more?
                        # But we must ensure pairwise coprime among added.
                        # This simple loop doesn't guarantee that.
                        pass
            
            # The above "add more" is flawed. Let's do a proper local search step.
            # Remove a. Then greedily add back best possible.
            # But that's expensive.
            
            # Alternative: Random restarts of greedy
            if random.random() < 0.1:
                break # Break out of swap loop to try a restart
            
        # If no simple improvement, try a random restart
        if not improved and random.random() < 0.05:
            # Randomized Greedy
            rand_chosen = []
            # Shuffle order slightly
            order = list(range(511, 0, -1))
            random.shuffle(order)
            # But we want large numbers to have higher priority
            # So we shuffle only the lower half?
            # Or just use a different exclusion set
            excl = set(random.sample(range(1, 512), 5))
            for a in range(511, 0, -1):
                if a in excl:
                    continue
                if is_coprime_set(rand_chosen, a):
                    rand_chosen.append(a)
            
            s = score(rand_chosen)
            if s > current_score:
                current_set = rand_chosen
                current_score = s
                improved = True
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 10:
                break

    if score(current_set) > best_score:
        best_set = current_set
        best_score = current_score

    return best_set

# EVOLVE-BLOCK-END
