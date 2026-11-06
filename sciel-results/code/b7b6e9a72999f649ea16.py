"""
Extended Sidon set construction using Simulated Annealing.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.
"""

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    
    # Initial valid sums for prescribed
    def get_initial_sums():
        sums = set()
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                sums.add(prescribed[i] + prescribed[j])
        return sums

    best_set = list(prescribed)
    best_size = len(best_set)
    
    # Helper to check validity and get sums
    def is_valid_and_sums(candidate_set):
        sums = set()
        sl = sorted(candidate_set)
        n = len(sl)
        for i in range(n):
            for j in range(i, n):
                s = sl[i] + sl[j]
                if s in sums:
                    return False, None
                sums.add(s)
        return True, sums

    # Greedy initialization to get a decent starting point
    def greedy_init():
        current_set = list(prescribed)
        current_sums = get_initial_sums()
        # Try adding smallest possible elements first
        for x in range(1, max_val + 1):
            if x in current_set:
                continue
            # Check if x can be added
            new_sums = {x + y for y in current_set} | {2 * x}
            if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
                current_set.append(x)
                current_sums.update(new_sums)
        return current_set

    # Run a few quick greedy inits to find a good start
    init_set = greedy_init()
    if len(init_set) > best_size:
        best_set = init_set
        best_size = len(init_set)
        
    # Simulated Annealing
    def simulate_annealing(start_set, time_limit):
        nonlocal best_set, best_size
        current_set = set(start_set)
        current_sums = set()
        valid, current_sums = is_valid_and_sums(current_set)
        if not valid:
            # Fallback to prescribed if init is bad (shouldn't happen)
            current_set = set(prescribed)
            current_sums = get_initial_sums()
            
        current_size = len(current_set)
        
        # Temperature parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.9995  # Cooling rate
        
        rng = random.Random()
        all_candidates = list(range(1, max_val + 1))
        
        # Pre-compute indices for prescribed to avoid removing them
        prescribed_list = list(prescribed_set)
        
        while time.time() < time_limit and T > T_min:
            # 1. Generate Neighbor
            move_type = rng.choice(['add', 'remove', 'swap'])
            new_set = set(current_set)
            
            if move_type == 'add':
                # Pick a random element not in current_set
                if current_size < max_val:
                    while True:
                        x = rng.choice(all_candidates)
                        if x not in current_set:
                            break
                    new_set.add(x)
                    
            elif move_type == 'remove':
                # Pick a random non-prescribed element to remove
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x = rng.choice(removable)
                    new_set.remove(x)
                    
            elif move_type == 'swap':
                # Pick a non-prescribed element to remove and a new one to add
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x_remove = rng.choice(removable)
                    # Pick x_add
                    while True:
                        x_add = rng.choice(all_candidates)
                        if x_add not in current_set:
                            break
                    new_set.remove(x_remove)
                    new_set.add(x_add)
            
            # If no valid move was generated (e.g., nothing removable), skip
            if new_set == current_set:
                continue
                
            # 2. Evaluate Neighbor
            valid_new, new_sums = is_valid_and_sums(new_set)
            
            if valid_new:
                new_size = len(new_set)
                delta = new_size - current_size
                
                # Acceptance probability
                if delta > 0:
                    # Always accept improvements
                    current_set = new_set
                    current_sums = new_sums
                    current_size = new_size
                    
                    # Update global best
                    if current_size > best_size:
                        best_set = list(current_set)
                        best_size = current_size
                        
                else:
                    # delta is 0 or negative (usually 0 since we only accept valid)
                    # Actually, if valid, size can only decrease if we removed without adding, 
                    # or stay same if swap/add-remove.
                    # We want to maximize size, so we prefer larger sets.
                    # If delta == 0, we might accept to escape local minima.
                    # If delta < 0, we accept with probability exp(-delta/T)
                    # Since we only care about size, delta < 0 means we lost an element.
                    # We should be reluctant to shrink unless it helps us grow later.
                    
                    if delta == 0:
                        # Random walk in same size layer
                        if rng.random() < 0.5: # Accept half the time to explore
                            current_set = new_set
                            current_sums = new_sums
                    else:
                        # delta < 0
                        prob = pow(2.718281828, -delta / T)
                        if rng.random() < prob:
                            current_set = new_set
                            current_sums = new_sums
                            current_size = new_size
            else:
                # Invalid neighbor, reject
                pass
            
            # 3. Cool down
            T *= alpha
            
        return current_size

    # Run multiple SA restarts
    # Use time budget wisely
    total_budget = deadline - time.time()
    num_restarts = 5
    time_per_restart = total_budget / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Create a perturbed start for the next SA run
        if i == 0:
            start_set = best_set
        else:
            # Perturb best_set by removing a random non-prescribed element
            # and letting SA re-optimize
            start_set = list(best_set)
            removable = [x for x in start_set if x not in prescribed_set]
            if removable:
                x_to_remove = random.choice(removable)
                start_set.remove(x_to_remove)
            
        remaining_time = deadline - time.time() - 5 # 5s safety
        if remaining_time <= 0:
            break
            
        # Run SA with a portion of the remaining time
        # We allow it to run until the specific restart time limit or global deadline
        simulate_annealing(start_set, time.time() + time_per_restart * 0.8)
        
        # After SA, do a quick local polish (greedy add)
        current_set = set(best_set)
        current_sums = set()
        valid, current_sums = is_valid_and_sums(current_set)
        
        # Try to add any remaining valid elements
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in range(1, max_val + 1):
                if x in current_set:
                    continue
                new_sums = {x + y for y in current_set} | {2 * x}
                if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
                    current_set.add(x)
                    current_sums.update(new_sums)
                    added = True
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(best_set)
                    break # Restart loop to check from beginning? Or just continue? 
                    # Restarting is safer to ensure smallest valid is picked if we want deterministic, 
                    # but for max size, order doesn't matter much.
        
    return best_set

if __name__ == "__main__":
    # Self-test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
