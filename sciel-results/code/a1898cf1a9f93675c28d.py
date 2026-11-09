import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # Bitset representation of reachable sums
    # bit i is 1 if sum i is reachable
    # We only care about sums up to 200
    
    def add_element(mask, x):
        return mask | (mask << x)
    
    def is_valid(mask):
        return ((mask >> 200) & 1) == 0

    def greedy_select(order, current_mask=1): # 1 means sum 0 is reachable
        selected = set()
        mask = current_mask
        for x in order:
            new_mask = add_element(mask, x)
            if is_valid(new_mask):
                selected.add(x)
                mask = new_mask
        return selected, mask

    # 1. Generate multiple greedy solutions with different orders
    candidates = []
    
    # Order 1: Descending
    order1 = list(range(100, 0, -1))
    sol1, mask1 = greedy_select(order1)
    candidates.append((sol1, mask1))
    
    # Order 2: Ascending
    order2 = list(range(1, 101))
    sol2, mask2 = greedy_select(order2)
    candidates.append((sol2, mask2))
    
    # Order 3: Sorted by value / (value+1) or some heuristic? 
    # Actually, let's try random shuffles to find better local optima
    for _ in range(50):
        if time.time() > deadline - 30.0:
            break
        random.shuffle(order1)
        sol, mask = greedy_select(order1)
        candidates.append((sol, mask))

    # Select the best initial candidate
    best_sol, best_mask = max(candidates, key=lambda x: len(x[0]))
    
    # 2. Local Search / Improvement
    # Strategy: 
    # a. Try to add any missing element.
    # b. Try 1-for-k swaps: remove one, try to add multiple.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Pass A: Greedy Add
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for x in range(1, 101):
                if x in best_sol:
                    continue
                new_mask = add_element(best_mask, x)
                if is_valid(new_mask):
                    best_sol.add(x)
                    best_mask = new_mask
                    changed = True
                    improved = True
        
        if time.time() > deadline - 10.0:
            break
            
        # Pass B: 1-for-k Swaps
        # For each element in best_sol, remove it, then try to add as many as possible
        for rem in list(best_sol):
            if time.time() > deadline - 5.0:
                break
            
            # Temporarily remove rem
            # We need to recompute the mask for best_sol - {rem}
            # This is expensive if done from scratch every time.
            # Optimization: Precompute masks for removal? 
            # Or just recompute from the set. 100 elements max.
            
            temp_sol = best_sol - {rem}
            temp_mask = 1
            for x in temp_sol:
                temp_mask = add_element(temp_mask, x)
                
            # Now try to add excluded elements to temp_sol
            added_any = False
            # Sort excluded elements? Maybe descending is better for cardinality?
            excluded = [x for x in range(1, 101) if x not in best_sol]
            # Try adding in descending order to maximize cardinality gain?
            for add in sorted(excluded, reverse=True):
                new_mask = add_element(temp_mask, add)
                if is_valid(new_mask):
                    temp_sol.add(add)
                    temp_mask = new_mask
                    added_any = True
            
            if len(temp_sol) > len(best_sol):
                best_sol = temp_sol
                best_mask = temp_mask
                improved = True
                # Restart the loop as the set changed significantly
                break 

    # 3. Final Polish: Randomized Hill Climbing
    # If we still have time, try random perturbations
    if time.time() < deadline - 5.0:
        current_best_size = len(best_sol)
        no_improve_count = 0
        while time.time() < deadline - 2.0 and no_improve_count < 100:
            # Perturb: remove 1-3 random elements
            perturb_size = random.randint(1, 3)
            perturb_elements = random.sample(list(best_sol), min(perturb_size, len(best_sol)))
            
            trial_sol = best_sol - set(perturb_elements)
            trial_mask = 1
            for x in trial_sol:
                trial_mask = add_element(trial_mask, x)
            
            # Greedy Add on the perturbed set
            for x in range(1, 101):
                if x in trial_sol:
                    continue
                new_mask = add_element(trial_mask, x)
                if is_valid(new_mask):
                    trial_sol.add(x)
                    trial_mask = new_mask
            
            if len(trial_sol) > current_best_size:
                best_sol = trial_sol
                best_mask = trial_mask
                current_best_size = len(trial_sol)
                no_improve_count = 0
            else:
                no_improve_count += 1

    return list(best_sol)
