import time
import random
from itertools import combinations
import numpy as np

# Pre-compute all 4-subsets of [1, 16] as bitmasks
ALL_SETS = []
for comb in combinations(range(16), 4):
    mask = 0
    for i in comb:
        mask |= (1 << i)
    ALL_SETS.append(mask)

N_SETS = len(ALL_SETS)
ALL_SETS_ARRAY = np.array(ALL_SETS, dtype=np.uint32)

def mask_to_list(mask):
    return [i + 1 for i in range(16) if mask & (1 << i)]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    # 1. Deterministic Greedy Baseline
    # We use a bitset to track which sets are in the current family for O(1) membership checks
    # But for the conflict check, we need to iterate over pairs.
    
    # Let's try a smarter initialization.
    # A random permutation greedy often performs well.
    
    best_family = []
    best_size = 0
    
    # We will perform multiple randomized greedy runs, followed by local search.
    
    num_greedy_runs = 20
    
    for run in range(num_greedy_runs):
        if time.time() > deadline - 10:
            break
            
        shuffled_indices = list(range(N_SETS))
        rng.shuffle(shuffled_indices)
        
        current_family = []
        family_set = set()
        
        for idx in shuffled_indices:
            if time.time() > deadline - 10:
                break
            if len(current_family) >= 250:
                break
                
            s_mask = ALL_SETS[idx]
            
            # Check validity
            valid = True
            n = len(current_family)
            # Optimization: Break early if conflict found
            for i in range(n):
                a = current_family[i]
                a_s = a & s_mask
                for j in range(i):
                    b = current_family[j]
                    if (a & b) == a_s == (b & s_mask):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_family.append(s_mask)
                family_set.add(s_mask)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            best_family_set = family_set.copy()
            
        # If we reached 250, we can stop early? 
        # The task says "at most 250". If we hit 250, that's the max possible value.
        if best_size >= 250:
            break

    # 2. Local Search to improve the best found so far
    # We try to add more sets or swap sets.
    
    if time.time() < deadline - 5:
        current_family = best_family[:]
        current_set = set(best_family)
        
        # First, try to add any remaining valid sets (greedy fill)
        # Iterate through all sets not in current_set
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            for idx in range(N_SETS):
                if time.time() > deadline - 5:
                    break
                if len(current_family) >= 250:
                    break
                    
                s_mask = ALL_SETS[idx]
                if s_mask in current_set:
                    continue
                
                valid = True
                n = len(current_family)
                for i in range(n):
                    a = current_family[i]
                    a_s = a & s_mask
                    for j in range(i):
                        b = current_family[j]
                        if (a & b) == a_s == (b & s_mask):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_family.append(s_mask)
                    current_set.add(s_mask)
                    added_any = True
                    if len(current_family) > best_size:
                        best_size = len(current_family)
                        best_family = current_family[:]
                        best_family_set = current_set.copy()
                        
        # Now, try swaps to potentially increase size further if we are stuck below 250
        # But since we did greedy fill, we are likely maximal (no single addition possible).
        # Swaps can help escape local optima.
        
        if len(current_family) < 250 and time.time() < deadline - 2:
            # Simulated annealing style swap
            # We want to maximize size.
            # Move: Remove one set, add two sets? No, standard swap is remove 1 add 1.
            # To increase size, we need to remove 1 and add 2? That's complex.
            # Usually, if we are maximal, we can't add. We must remove to free up space.
            # However, if we remove 1 and add 1, size stays same.
            # If we remove 1 and add 2, size increases by 1.
            # Let's try: Remove a random set, then try to add TWO sets that are now valid.
            
            max_swap_attempts = 1000
            for _ in range(max_swap_attempts):
                if time.time() > deadline - 2:
                    break
                if len(current_family) >= 250:
                    break
                    
                # Pick a random set to remove
                rem_idx = rng.randint(0, len(current_family) - 1)
                rem_mask = current_family[rem_idx]
                
                # Remove it
                current_family.pop(rem_idx)
                current_set.discard(rem_mask)
                
                # Try to add two sets
                added_count = 0
                # We need to find two sets that can be added sequentially
                # To save time, we just iterate through random candidates
                
                # Candidate pool: sets not in current_set
                # We'll just pick random ones from ALL_SETS
                
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    if added_count >= 2:
                        break
                        
                    idx = rng.randint(0, N_SETS - 1)
                    s_mask = ALL_SETS[idx]
                    if s_mask in current_set:
                        continue
                        
                    valid = True
                    n = len(current_family)
                    for i in range(n):
                        a = current_family[i]
                        a_s = a & s_mask
                        for j in range(i):
                            b = current_family[j]
                            if (a & b) == a_s == (b & s_mask):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        current_family.append(s_mask)
                        current_set.add(s_mask)
                        added_count += 1
                        
                # If we added at least 1, we might have increased size or kept it same
                # If we added 2, size increased by 1 (removed 1, added 2)
                # If we added 1, size stayed same (removed 1, added 1)
                # If we added 0, size decreased by 1. We should probably revert if 0?
                # But if we reverted, we are back to same state.
                
                if added_count > 0:
                    if len(current_family) > best_size:
                        best_size = len(current_family)
                        best_family = current_family[:]
                        best_family_set = current_set.copy()
                else:
                    # Revert if nothing was added
                    current_family.append(rem_mask)
                    current_set.add(rem_mask)
                    
    # Final check
    if len(current_family) > len(best_family):
        best_family = current_family
        
    # Convert best_family masks back to sorted lists
    return [mask_to_list(m) for m in best_family]
