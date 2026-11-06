# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def _greedy_numpy(order, counts_arr):
        chosen = []
        for x in order:
            if counts_arr[2 * x] >= 2:
                continue
            valid = True
            for y in chosen:
                s = x + y
                if counts_arr[s] >= 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts_arr[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    counts_arr[s] += 1
        return chosen

    # 1. Deterministic Greedy Baseline (ensures score >= 1.0)
    order1 = list(range(1, 1001))
    counts1 = np.zeros(2001, dtype=np.int32)
    best_set = _greedy_numpy(order1, counts1)
    best_size = len(best_set)

    # 2. Structural Change: Local Search with INSERT, REMOVE, and SWAP moves
    # The seed program's local search (if any) was effectively capped at greedy size.
    # This implementation explicitly allows growing the set (INSERT) and shrinking it (REMOVE)
    # to escape local minima, plus SWAP for fine-tuning.
    
    # We maintain:
    # - mask: binary array of length 1000
    # - active: list of 1-based integers currently in the set
    # - counts: array of sum counts (size 2001)
    # - penalty: sum of max(0, counts[s] - 2) for s in 2..2000
    
    def _init_state(mask):
        active = list(np.nonzero(mask)[0] + 1)
        counts = np.zeros(2001, dtype=np.int32)
        for i, x in enumerate(active):
            counts[2*x] += 1
            for y in active[i+1:]:
                counts[x+y] += 1
        penalty = np.sum(np.maximum(0, counts[2:2001] - 2))
        return active, counts, int(penalty)

    def _calc_penalty(counts):
        return int(np.sum(np.maximum(0, counts[2:2001] - 2)))

    # SA parameters
    T_start = 20.0
    T_end = 0.01
    # We will run multiple restarts.
    
    rng = np.random.default_rng(12345)
    
    # Start from the greedy solution
    base_mask = np.zeros(1000, dtype=np.int8)
    for x in best_set:
        base_mask[x-1] = 1
        
    # We will try to improve upon this.
    # Strategy: Run SA with a time budget.
    # To allow exploration, we start from the greedy set but allow "bad" moves (penalty increases) 
    # to escape local optima where we might be stuck with a suboptimal valid set.
    
    # We can also try starting from a random set to see if we find a better basin.
    # But greedy is a strong baseline. Let's stick to refining the greedy solution first, 
    # and maybe do a few random restarts if time permits.
    
    # Actually, a "random restart" from scratch might find a completely different structure.
    # However, B2[2] sets are likely to have similar structures.
    # Let's focus on the local search around the greedy solution with INSERT/REMOVE/SWAP.
    
    def _run_local_search(initial_mask, time_limit):
        mask = initial_mask.copy()
        active, counts, penalty = _init_state(mask)
        current_size = len(active)
        
        best_mask = mask.copy()
        best_size = current_size
        best_penalty = penalty
        
        T = T_start
        # Cooling factor to reach T_end in roughly the time budget
        # We don't know the number of steps in advance, so we use a fixed cooling rate per step
        # and check time.
        cooling = 0.9995
        
        while time.time() < time_limit:
            # Choose move type
            # Probabilities: 50% Insert, 30% Remove, 20% Swap?
            # Or just pick a random action.
            r = rng.random()
            
            if r < 0.4:
                # INSERT: pick a random x not in mask
                # Find a random 0 in mask
                # Instead of scanning, pick random idx and check
                for _ in range(10): # Try a few times to find a 0
                    idx = rng.integers(0, 1000)
                    if mask[idx] == 0:
                        x = idx + 1
                        break
                else:
                    continue # All 1s? Unlikely.
                
                # Calculate delta
                # Adding x:
                # counts[2x] += 1
                # counts[x+y] += 1 for y in active
                # Penalty change:
                # For each sum s affected, if counts[s] was 2, it becomes 3 -> penalty +1
                # if counts[s] was 3, it becomes 4 -> penalty +1
                # Generally, penalty increases by the number of sums s such that counts[s] >= 2.
                # Let's just compute new penalty.
                
                # Optimization: Instead of copying counts, we can compute delta_penalty directly.
                # delta_penalty = 0
                # if counts[2*x] >= 2: delta_penalty += 1
                # for y in active:
                #     s = x+y
                #     if counts[s] >= 2: delta_penalty += 1
                
                delta_penalty = 0
                if counts[2*x] >= 2:
                    delta_penalty += 1
                for y in active:
                    s = x + y
                    if counts[s] >= 2:
                        delta_penalty += 1
                
                new_size = current_size + 1
                new_penalty = penalty + delta_penalty
                
                # Score: Size - Penalty * Weight
                # Weight should be high enough to prefer valid sets.
                # If we want to maximize size subject to penalty=0, we can use a large weight.
                # But to allow exploration, we use a moderate weight.
                # Let's use Weight = 10.
                weight = 10.0
                old_score = current_size - penalty * weight
                new_score = new_size - new_penalty * weight
                delta_score = new_score - old_score
                
                if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                    # Accept
                    mask[idx] = 1
                    active.append(x)
                    counts[2*x] += 1
                    for y in active[:-1]: # active[:-1] excludes the newly added x? No, we appended.
                        # Wait, active was updated. We need to update counts for y in OLD active.
                        # Let's be careful.
                        pass # This logic is flawed in-place. Let's restore and do it properly or use a temp array.
                    
                    # Better: Apply change to counts, check, if reject restore.
                    # Since we already calculated delta_penalty, we just need to update counts if accepted.
                    # But we modified `active` and `mask` before checking?
                    # Let's revert that style. Calculate, decide, then update.
                    
                    # ROLLBACK for this attempt to keep code clean:
                    mask[idx] = 0
                    active.pop() # Remove the x we added temporarily? No, we didn't add it yet in the "decide" phase.
                    # Actually, I appended x to active in the "Accept" block above.
                    # Let's restructure:
                    # 1. Calculate delta
                    # 2. Decide
                    # 3. If accept: Update state
                    # 4. If reject: Do nothing
                    
                    # So, if we are in the "if accept" block, we should update.
                    # But I need to make sure I didn't modify state before the check.
                    # In the code above, I did `active.append(x)` inside the `if`.
                    # But I also need to update `counts`.
                    
                    # Let's redo the update properly.
                    # The `delta_penalty` calculation is correct.
                    # If accepted:
                    counts[2*x] += 1
                    for y in active[:-1]: # active now has x at the end. active[:-1] is old active.
                        counts[x+y] += 1
                    # Wait, `active` was appended to BEFORE the counts update?
                    # In my code:
                    # if accept:
                    #    mask[idx] = 1
                    #    active.append(x)
                    #    ...
                    # So `active` contains x. `active[:-1]` is the old set.
                    # This is correct.
                    
                    current_size = new_size
                    penalty = new_penalty
                    
                    if penalty == 0 and current_size > best_size:
                        best_size = current_size
                        best_mask = mask.copy()
                        
            elif r < 0.7:
                # REMOVE: pick a random x in mask
                if current_size == 0:
                    continue
                idx_in_active = rng.integers(0, current_size)
                x = active[idx_in_active]
                idx = x - 1
                
                # Calculate delta
                # Removing x:
                # counts[2x] -= 1
                # counts[x+y] -= 1 for y in active (y != x)
                # Penalty change:
                # For each sum s affected, if counts[s] was 3, it becomes 2 -> penalty -1
                # if counts[s] was 4, it becomes 3 -> penalty -1
                # Generally, penalty decreases by the number of sums s such that counts[s] > 2.
                
                delta_penalty = 0
                if counts[2*x] > 2:
                    delta_penalty -= 1
                for y in active:
                    if y != x:
                        s = x + y
                        if counts[s] > 2:
                            delta_penalty -= 1
                
                new_size = current_size - 1
                new_penalty = penalty + delta_penalty
                
                weight = 10.0
                old_score = current_size - penalty * weight
                new_score = new_size - new_penalty * weight
                delta_score = new_score - old_score
                
                if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                    # Accept
                    mask[idx] = 0
                    # Remove x from active
                    # active is a list. Removal is O(N).
                    # We can swap with last and pop for O(1).
                    active[idx_in_active] = active[-1]
                    active.pop()
                    
                    # Update counts
                    counts[2*x] -= 1
                    for y in active:
                        counts[x+y] -= 1
                    
                    current_size = new_size
                    penalty = new_penalty
                    
                    if penalty == 0 and current_size > best_size:
                        best_size = current_size
                        best_mask = mask.copy()
                        
            else:
                # SWAP: pick x in mask, y not in mask
                if current_size == 0 or current_size == 1000:
                    continue
                
                # Pick x
                idx_in_active = rng.integers(0, current_size)
                x = active[idx_in_active]
                
                # Pick y
                for _ in range(10):
                    idx_y = rng.integers(0, 1000)
                    if mask[idx_y] == 0:
                        y = idx_y + 1
                        break
                else:
                    continue
                
                if x == y:
                    continue
                
                # Calculate delta
                # Remove x, Add y.
                # This is equivalent to two moves, but we can combine.
                # Let's just calculate new penalty by simulating.
                # Since N is small (1000), and active is ~500, this is O(N).
                
                # Temporarily remove x and add y to a copy of counts? 
                # Or just calculate delta.
                # Delta from removing x:
                #   -1 for each s in {2x, x+z (z!=x)} where counts[s] > 2
                # Delta from adding y:
                #   +1 for each s in {2y, y+w (w in active \ {x} + {y}? No, active changes)} 
                #   Careful: The set of partners changes.
                
                # Easier: Just compute the new penalty from scratch?
                # O(N^2) is 250k ops. If we do this for every swap, it might be slow.
                # But swaps are 30% of moves.
                # Let's try to be smarter.
                
                # Actually, let's just do two steps: Remove x, then Add y.
                # But we need the intermediate state.
                # Let's just calculate the penalty change directly.
                
                # Sums affected by removing x:
                # 2x, and x+z for z in active, z!=x.
                # Sums affected by adding y:
                # 2y, and y+w for w in (active \ {x}) U {y}.
                # Note: y is not in active yet.
                
                # Let's just use a temporary counts array for this move?
                # Copying counts (2001 ints) is fast.
                temp_counts = counts.copy()
                
                # Remove x
                temp_counts[2*x] -= 1
                for z in active:
                    if z != x:
                        temp_counts[x+z] -= 1
                
                # Add y
                temp_counts[2*y] += 1
                for w in active:
                    if w != x:
                        temp_counts[y+w] += 1
                
                new_penalty = _calc_penalty(temp_counts)
                new_size = current_size # Size unchanged
                
                weight = 10.0
                old_score = current_size - penalty * weight
                new_score = new_size - new_penalty * weight
                delta_score = new_score - old_score
                
                if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                    # Accept
                    # Update mask
                    mask[x-1] = 0
                    mask[y-1] = 1
                    
                    # Update active
                    active[idx_in_active] = y
                    
                    # Update counts
                    counts = temp_counts
                    penalty = new_penalty
                    # current_size unchanged
                    
                    if penalty == 0 and current_size > best_size:
                        best_size = current_size
                        best_mask = mask.copy()
                        
            T *= cooling
            if T < T_end:
                break
        
        return best_mask, best_size

    # Run local search
    # We can run it once for the full budget, or multiple shorter runs.
    # Multiple shorter runs with different perturbations might be better.
    
    # Let's do 5 runs, each with 1/5 of the time.
    num_runs = 5
    time_per_run = (deadline - time.time()) / num_runs
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
            
        # Perturb the best solution found so far
        current_best_mask = np.zeros(1000, dtype=np.int8)
        if best_size > 0:
            # We need to reconstruct the mask from best_set or keep track of best_mask
            # Let's keep track of best_mask in the outer scope or return it.
            # The _run_local_search returns best_mask.
            # For the first run, we use base_mask.
            # For subsequent runs, we use the best mask found.
            if i == 0:
                start_mask = base_mask
            else:
                start_mask = best_mask_found # Need to store this
                
        else:
            start_mask = base_mask
            
        # Actually, let's just store the best mask found in a variable outside the loop.
        # Let's restructure.
        
        if i == 0:
            start_mask = base_mask
        else:
            start_mask = best_mask_global
            
        # Add some noise to escape local minima
        perturbed_mask = start_mask.copy()
        num_flips = 10
        for _ in range(num_flips):
            idx = rng.integers(0, 1000)
            perturbed_mask[idx] = 1 - perturbed_mask[idx]
            
        time_limit = time.time() + time_per_run
        run_mask, run_size = _run_local_search(perturbed_mask, time_limit)
        
        if run_size > best_size:
            best_size = run_size
            best_mask_global = run_mask
            best_set = list(np.nonzero(run_mask)[0] + 1)
        else:
            # Keep the previous best
            if i == 0:
                best_mask_global = start_mask
            else:
                best_mask_global = best_mask_global # Unchanged
                
    # If no improvement, return the original greedy set
    # But we need to make sure best_set corresponds to best_size
    # If best_size was updated, best_set is updated.
    # If not, best_set is still the greedy one.
    
    return best_set
# EVOLVE-BLOCK-END
