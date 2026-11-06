import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    # Elements are 1..16, so we use bits 0..15
    all_sets = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Map mask to original sorted list for output
    mask_to_list = {}
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        mask_to_list[mask] = sorted(comb)

    def is_valid_addition(cand, family):
        """
        Check if cand can be added to family without creating a sunflower.
        Sunflower: 3 distinct sets A, B, C with A&B = A&C = B&C.
        We check if adding cand creates such a triple with any two existing sets.
        
        Optimization: Group existing sets by their intersection with cand.
        If two existing sets A, B have the same intersection S with cand,
        then we must check if A & B == S. If so, (A, B, cand) is a sunflower.
        """
        groups = {}
        for existing in family:
            inter = cand & existing
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(existing)
        
        for S, members in groups.items():
            if len(members) < 2:
                continue
            # Check pairs within this group
            n = len(members)
            for i in range(n):
                mi = members[i]
                for j in range(i + 1, n):
                    mj = members[j]
                    if (mi & mj) == S:
                        return False
        return True

    # Initial Solution: Greedy
    def greedy_init(rng):
        family = []
        family_set = set()
        order = list(range(n_sets))
        rng.shuffle(order)
        
        for idx in order:
            cand = all_sets[idx]
            if cand in family_set:
                continue
            if is_valid_addition(cand, family):
                family.append(cand)
                family_set.add(cand)
        return family

    best_family = []
    best_size = 0
    rng = random.Random(42)
    
    # Run a few greedy inits to find a good starting point
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        fam = greedy_init(rng)
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam[:]

    # Simulated Annealing
    # State: family (list of masks)
    # Move: Remove a random element, try to add a random element not in family
    # Energy: -len(family)
    
    current_family = best_family[:]
    current_size = len(current_family)
    current_energy = -current_size
    
    best_family = current_family[:]
    best_size = current_size
    
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    steps = 0
    last_improve_time = start_time
    
    while time.time() < deadline - 5:
        steps += 1
        
        # Check for stagnation and restart if necessary? 
        # For now, just let SA run.
        
        if len(current_family) == 0:
            break
            
        # Pick a random set to remove
        remove_idx = rng.randint(0, len(current_family) - 1)
        removed_set = current_family[remove_idx]
        
        # Pick a random set to add
        # We want to try sets that are not currently in the family
        # To avoid O(N) check for membership, we can just pick a random index 
        # and check if it's in the family set.
        
        # For efficiency, maintain a set of current family masks
        # But current_family is a list. Let's create a set for quick lookup.
        # This is O(N) per step if we rebuild it. 
        # Better: keep a set `current_set` updated alongside `current_family`.
        
        # Let's refactor to keep a set.
        # Re-initializing the loop structure to be cleaner with a set.
        
        # Actually, let's just use the list and a set variable updated incrementally.
        # But the current code doesn't have a set variable in the outer scope for SA.
        # Let's initialize it.
        
        # Break out of this specific while loop structure and rewrite the SA part
        # to be cleaner.
        
        if steps == 1:
            # Initialize SA state properly
            current_family_set = set(current_family)
            current_size = len(current_family)
            best_family = current_family[:]
            best_size = current_size
            best_family_set = set(best_family)
            
        # Generate move
        remove_idx = rng.randint(0, len(current_family) - 1)
        removed_set = current_family[remove_idx]
        
        # Try to find a valid addition
        # We try a few random candidates for the addition
        added_set = None
        for _ in range(10):
            cand_idx = rng.randint(0, n_sets - 1)
            cand = all_sets[cand_idx]
            if cand in current_family_set:
                continue
            
            # Check if cand is valid with (current_family - {removed_set})
            # We need to check cand against all other sets in current_family except removed_set
            # This is expensive. 
            # Optimization: 
            # If we remove R, we only need to check if Cand is compatible with F \ {R}.
            # This is still O(N^2) in worst case.
            
            # Let's just do the full check on the modified family.
            # To make it faster, we can check if Cand is compatible with F first.
            # If it is compatible with F, it is definitely compatible with F \ {R}.
            # If it is NOT compatible with F, it might still be compatible with F \ {R} 
            # if the conflict was with R.
            
            # So:
            # 1. Check if Cand is valid in F.
            #    If yes, it's a great move (size stays same, or we can swap).
            #    Actually, if we remove R and add Cand, and Cand was invalid in F 
            #    only because of R, then this is a valid move.
            
            # Let's just construct the new family and check validity? 
            # No, that's too slow.
            
            # Let's use the property:
            # Cand is valid in F \ {R} iff there is no pair A, B in F \ {R} 
            # such that A&B = A&Cand = B&Cand.
            
            # We can check this by iterating through F \ {R}.
            
            # Let's implement a helper `is_valid_in_subset(cand, family, exclude_idx)`
            
            # For now, let's just do a simple local search: 
            # 1. Remove R.
            # 2. Try to add C.
            # 3. If valid, accept.
            
            # We will check validity against the list `current_family` but skipping `removed_set`.
            
            valid = True
            # Grouping check for cand against current_family excluding removed_set
            groups = {}
            for i, existing in enumerate(current_family):
                if i == remove_idx:
                    continue
                inter = cand & existing
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(existing)
            
            for S, members in groups.items():
                if len(members) < 2:
                    continue
                n = len(members)
                for i in range(n):
                    mi = members[i]
                    for j in range(i + 1, n):
                        mj = members[j]
                        if (mi & mj) == S:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                added_set = cand
                break
        
        if added_set is None:
            # No valid addition found in this trial
            # Maybe try to just remove? No, that decreases size.
            # Continue to next step
            T *= cooling_rate
            continue
        
        # Proposed move: Remove removed_set, Add added_set
        # New size is same.
        # We accept if it leads to a "better" state? 
        # Since size is same, we need a different metric or just random walk.
        # To escape local optima, we should allow moves that keep size same 
        # but change composition, hoping to enable future growth.
        # However, if we only do swaps, we never grow.
        
        # Better Move Strategy:
        # 1. Try to ADD a set (grow). If valid, accept.
        # 2. Try to SWAP a set (remove one, add one). If valid, accept with prob exp(-deltaE/T) where deltaE=0.
        # 3. Try to REMOVE a set (shrink). If valid (always is), accept with prob exp(-deltaE/T) where deltaE=1 (worse).
        
        # Let's restructure the loop to try ADD first.
        
        # --- RESTRUCTURED SA STEP ---
        
        # 1. Try to add a random set
        improved = False
        for _ in range(5):
            cand_idx = rng.randint(0, n_sets - 1)
            cand = all_sets[cand_idx]
            if cand in current_family_set:
                continue
            if is_valid_addition(cand, current_family):
                # Accept growth
                current_family.append(cand)
                current_family_set.add(cand)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_family = current_family[:]
                    best_family_set = current_family_set.copy()
                    last_improve_time = time.time()
                improved = True
                break
        
        if improved:
            T = max(min_T, T * 0.99) # Reset or cool slightly
            continue
            
        # 2. Try to swap
        # Remove random, add random
        remove_idx = rng.randint(0, len(current_family) - 1)
        removed_set = current_family[remove_idx]
        
        swap_success = False
        for _ in range(5):
            cand_idx = rng.randint(0, n_sets - 1)
            cand = all_sets[cand_idx]
            if cand in current_family_set:
                continue
            
            # Check if cand is valid in F \ {removed_set}
            # We reuse the logic from above
            groups = {}
            for i, existing in enumerate(current_family):
                if i == remove_idx:
                    continue
                inter = cand & existing
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(existing)
            
            valid = True
            for S, members in groups.items():
                if len(members) < 2:
                    continue
                n = len(members)
                for i in range(n):
                    mi = members[i]
                    for j in range(i + 1, n):
                        mj = members[j]
                        if (mi & mj) == S:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Perform swap
                current_family[remove_idx] = cand
                current_family_set.remove(removed_set)
                current_family_set.add(cand)
                swap_success = True
                break
        
        if swap_success:
            # Accept swap (energy change 0)
            T *= cooling_rate
            continue
            
        # 3. Try to remove (shrink) to escape local optima
        # This is risky, only do if T is high or we are stuck
        if T > 0.5 and rng.random() < 0.1:
            remove_idx = rng.randint(0, len(current_family) - 1)
            removed_set = current_family[remove_idx]
            current_family.pop(remove_idx)
            current_family_set.remove(removed_set)
            current_size -= 1
            T *= cooling_rate
            continue

        T *= cooling_rate
        
    # Convert best_family to sorted lists
    result = []
    for m in best_family:
        result.append(mask_to_list[m])
    
    return result
