import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 120.0  # Leave margin for 150s limit
    
    N = 1000
    max_sum = 2 * N
    
    # Helper to check validity and update counts
    # We will maintain a list of elements and a dict of sum counts
    
    best_size = 0
    best_set = []
    
    def get_initial_set():
        # Greedy construction with random ordering
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        chosen = []
        counts = {}
        
        for x in candidates:
            # Check if adding x is valid
            # Sums involving x: x + y for y in chosen, and 2*x
            valid = True
            new_sums = []
            for y in chosen:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            # Check counts
            # We need to simulate the addition
            # If any sum s in new_sums has count[s] + (multiplicity in new_sums) > 2, invalid
            # Note: if x+y1 == x+y2, then y1==y2, so no duplicate sums from x+chosen
            # unless y1==y2 which is impossible as chosen has distinct elements.
            # So each s in new_sums appears exactly once in the "new" part, except 2*x is separate.
            # Wait, if x + y == 2*x, then y == x, impossible since y in chosen, x not in chosen.
            # So all new sums are distinct from each other?
            # x + y1 == x + y2 => y1 == y2.
            # So yes, distinct.
            
            # We just need to check if counts[s] + 1 > 2 for any s in new_sums
            # i.e., counts[s] >= 2
            for s in new_sums:
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        return chosen, counts

    def remove_element(chosen, counts, x):
        # Remove x from chosen and update counts
        # Sums to decrement: x + y for y in chosen (excluding x), and 2*x
        # We need to be careful: we are removing x, so we iterate over the *current* chosen
        # But x is in chosen.
        # Let's make a copy of chosen to iterate? Or just filter.
        # Since size is small, copying is fine.
        chosen_copy = chosen[:]
        chosen_copy.remove(x)
        
        sums_to_dec = [x + y for y in chosen_copy]
        sums_to_dec.append(2 * x)
        
        for s in sums_to_dec:
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        
        return chosen_copy, counts

    def try_add_element(chosen, counts, x):
        # Check if x can be added
        # x must not be in chosen
        if x in chosen:
            return False, chosen, counts
            
        new_sums = [x + y for y in chosen]
        new_sums.append(2 * x)
        
        # Check validity
        for s in new_sums:
            if counts.get(s, 0) >= 2:
                return False, chosen, counts
        
        # Update
        new_chosen = chosen + [x]
        new_counts = counts.copy() # Shallow copy is fine as we only add to new keys or increment existing
        # Actually, we should update in place or return new state.
        # To avoid mutation issues, let's return new state.
        # But copying dict is O(|counts|). |counts| is ~ 2*|chosen|^2? No, |counts| is number of distinct sums, ~ |chosen|^2/2.
        # For |chosen|=50, ~1250 entries. Copying is fast.
        
        for s in new_sums:
            new_counts[s] = new_counts.get(s, 0) + 1
            
        return True, new_chosen, new_counts

    # Main loop
    # We will run multiple restarts and local improvements
    
    # Initial greedy sets
    for i in range(20): # Try 20 random greedy starts
        if time.time() - start_time > time_limit:
            break
        chosen, counts = get_initial_set()
        current_size = len(chosen)
        if current_size > best_size:
            best_size = current_size
            best_set = chosen[:]
        
        # Local Search: Perturb and Repair
        # Try to improve the current set
        for step in range(100): # 100 steps of perturbation
            if time.time() - start_time > time_limit:
                break
            
            # Remove k random elements
            k = random.randint(3, 8)
            if current_size <= k:
                continue
                
            elements_to_remove = random.sample(chosen, k)
            
            # Perform removals
            temp_chosen = chosen[:]
            temp_counts = counts.copy()
            
            for x in elements_to_remove:
                temp_chosen, temp_counts = remove_element(temp_chosen, temp_counts, x)
            
            # Now try to add elements back, prioritizing those that were removed?
            # Or just greedily add from the full candidate list?
            # Greedily adding from full list is expensive (1000 candidates * check).
            # But check is O(|S|). 1000 * 50 = 50k ops. Fast.
            
            # Let's try to add back the removed ones first, then others.
            # Actually, just iterate through all candidates 1..1000 and add if valid.
            # This will find a maximal set again.
            
            # To make it faster, we can just try to add the removed ones, 
            # and then fill with any other available.
            
            # Strategy: 
            # 1. Try to add back removed elements in random order.
            # 2. Then iterate through all 1..1000 and add if valid.
            
            random.shuffle(elements_to_remove)
            for x in elements_to_remove:
                ok, temp_chosen, temp_counts = try_add_element(temp_chosen, temp_counts, x)
                if ok:
                    pass # already updated
            
            # Now fill with any other element
            candidates = [x for x in range(1, N+1) if x not in temp_chosen]
            random.shuffle(candidates)
            
            for x in candidates:
                ok, temp_chosen, temp_counts = try_add_element(temp_chosen, temp_counts, x)
                if ok:
                    pass # already updated
            
            new_size = len(temp_chosen)
            if new_size > current_size:
                chosen = temp_chosen
                counts = temp_counts
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = chosen[:]
            else:
                # If we didn't improve, revert? 
                # Or keep the new maximal set? 
                # Keeping it allows further perturbation from a different point.
                # But if size is smaller, we lose progress.
                # Let's revert if size decreased, to maintain a "best so far" trajectory?
                # No, hill climbing on size usually requires accepting equal or better.
                # If we accept worse, it's simulated annealing.
                # Let's just keep the best valid set found in this restart.
                # Actually, if new_size < current_size, we should probably revert to chosen/counts
                # so we don't degrade.
                # But maybe the degraded set has more "room"?
                # Let's stick to: if new_size >= current_size, accept. Else revert.
                pass # We already have chosen/counts from before perturbation

    return best_set

# EVOLVE-BLOCK-END
