"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from math import gcd

    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill

    # Denominators are 2,3,...,48 -> indices 0..46
    n = 47
    denominators = list(range(2, 49))

    # Compute LCM of 2..48 for exact integer arithmetic
    def lcm(a, b):
        return a // gcd(a, b) * b

    L = 1
    for d in denominators:
        L = lcm(L, d)

    # Integer weights: weight[i] = L // denominators[i]
    weights = [L // d for d in denominators]
    target = L  # We need sum of weights < target

    # Initial Greedy Solution
    # Sort by weight descending (denominator ascending)
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    current_set = set()
    current_sum = 0
    for i in order:
        if current_sum + weights[i] < target:
            current_set.add(i)
            current_sum += weights[i]
    
    best_set = set(current_set)
    best_sum = current_sum
    best_deficit = target - best_sum

    # Simulated Annealing
    # Parameters
    initial_temp = 1000.0
    final_temp = 1.0
    cooling_rate = 0.999
    
    # We will run SA until time runs out
    # To make it robust, we run multiple restarts if time permits, 
    # but usually one long SA run with good initialization is better.
    # However, to escape deep local minima, restarts help.
    # Let's do a hybrid: run SA, if stuck (no improvement for X steps), restart with perturbation.
    
    rng = random.Random(42)
    
    # Precompute list of indices for random access
    indices = list(range(n))
    
    # To speed up random selection of unselected/selected, 
    # we can just pick a random index and check membership.
    
    steps_since_improvement = 0
    max_stagnation = 10000 # If no improvement for this many steps, restart
    
    temp = initial_temp
    
    while time.time() < deadline:
        if steps_since_improvement > max_stagnation:
            # Restart with perturbation
            # Take best_set, remove a random few, add random few valid ones
            current_set = set(best_set)
            current_sum = best_sum
            
            # Perturb: remove 3 random items
            if len(current_set) > 3:
                to_remove = rng.sample(list(current_set), 3)
                for i in to_remove:
                    current_set.discard(i)
                    current_sum -= weights[i]
            
            # Add 3 random items if they fit
            for _ in range(3):
                j = rng.choice(indices)
                if j not in current_set and current_sum + weights[j] < target:
                    current_set.add(j)
                    current_sum += weights[j]
            
            temp = initial_temp * 0.5 # Restart with lower temp? Or high?
            # High temp to explore
            temp = initial_temp
            steps_since_improvement = 0
            continue
            
        # Generate a move
        # Options: Add, Remove, Swap
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add a random unselected item
            j = rng.choice(indices)
            if j in current_set:
                continue
            new_sum = current_sum + weights[j]
            if new_sum < target:
                current_set.add(j)
                current_sum = new_sum
                new_deficit = target - current_sum
                
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    best_set = set(current_set)
                    best_sum = current_sum
                    steps_since_improvement = 0
                else:
                    # Check acceptance of worse move
                    delta = new_deficit - (target - (current_sum - weights[j])) # This is just weights[j]
                    # Actually, delta E = New Deficit - Old Deficit
                    old_deficit = target - (current_sum - weights[j])
                    delta = new_deficit - old_deficit
                    if delta > 0:
                        p = math.exp(-delta / temp)
                        if rng.random() < p:
                            pass # Accept
                        else:
                            # Reject
                            current_set.discard(j)
                            current_sum -= weights[j]
                    # If delta <= 0, it was an improvement, already handled
            # If new_sum >= target, move is invalid, skip (do nothing)
            
        elif move_type < 0.6:
            # Remove a random selected item
            if not current_set:
                continue
            i = rng.choice(list(current_set))
            current_set.discard(i)
            current_sum -= weights[i]
            new_deficit = target - current_sum
            
            # Removal always increases deficit (worsens), so we only accept if it helps escape a trap?
            # In SA for minimization, we accept worse moves.
            # But here, removing an item *always* makes the deficit larger (sum smaller).
            # So this is a "worse" move in terms of objective.
            old_deficit = target - (current_sum + weights[i])
            delta = new_deficit - old_deficit # Positive
            if delta > 0:
                p = math.exp(-delta / temp)
                if rng.random() < p:
                    # Accept
                    steps_since_improvement += 1
                else:
                    # Reject
                    current_set.add(i)
                    current_sum += weights[i]
            else:
                # This case shouldn't happen for removal
                steps_since_improvement += 1
            
        else:
            # Swap: Remove one, Add one
            if not current_set:
                continue
            i = rng.choice(list(current_set))
            j = rng.choice(indices)
            if i == j:
                continue
            if j in current_set:
                continue
            
            new_sum = current_sum - weights[i] + weights[j]
            if new_sum < target:
                current_set.discard(i)
                current_set.add(j)
                current_sum = new_sum
                new_deficit = target - current_sum
                
                old_deficit = target - (current_sum - weights[j] + weights[i])
                delta = new_deficit - old_deficit
                
                if delta <= 0:
                    # Improvement
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        best_set = set(current_set)
                        best_sum = current_sum
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                else:
                    # Worse
                    p = math.exp(-delta / temp)
                    if rng.random() < p:
                        steps_since_improvement += 1
                    else:
                        # Reject
                        current_set.discard(j)
                        current_set.add(i)
                        current_sum = current_sum - weights[j] + weights[i]
                        steps_since_improvement += 1
            
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp

    # Final Verification
    assert len(best_set) > 0, "Must select at least one item"
    total = sum(weights[i] for i in best_set)
    assert total < target, f"Sum {total} not strictly below target {target}"

    return list(best_set)
# EVOLVE-BLOCK-END
