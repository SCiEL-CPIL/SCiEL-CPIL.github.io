import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    mask_to_idx = {}
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        mask_to_idx[mask] = len(all_sets) - 1
        
    total = len(all_sets)
    
    # Helper to get conflicts for a pair of masks
    def get_conflicts(mask_a, mask_b):
        """
        Returns a list of indices of sets S such that 
        A, B, S form a sunflower.
        """
        core = mask_a & mask_b
        core_size = bin(core).count('1')
        union = mask_a | mask_b
        # Elements not in A or B
        available_mask = ((1 << n) - 1) & ~union
        
        # We need to pick (k - core_size) elements from available_mask
        num_to_pick = k - core_size
        if num_to_pick < 0:
            return []
        
        # Get list of available elements
        avail_elems = []
        for i in range(n):
            if available_mask & (1 << i):
                avail_elems.append(i)
                
        if len(avail_elems) < num_to_pick:
            return []
            
        conflicts = []
        for pick in combinations(avail_elems, num_to_pick):
            s_mask = core
            for x in pick:
                s_mask |= (1 << x)
            if s_mask in mask_to_idx:
                conflicts.append(mask_to_idx[s_mask])
        return conflicts

    # Greedy Initialization
    rng = random.Random(42)
    order = list(range(total))
    rng.shuffle(order)
    
    family = []
    in_family = [False] * total
    conflict_count = [0] * total
    
    for idx in order:
        if time.time() > deadline - 10:
            break
        if conflict_count[idx] == 0:
            # Add idx to family
            family.append(idx)
            in_family[idx] = True
            # Update conflict counts for all existing members
            for j in family[:-1]:
                conflicts = get_conflicts(all_sets[idx], all_sets[j])
                for c in conflicts:
                    conflict_count[c] += 1

    best_family = family[:]
    best_size = len(family)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    while T > min_T and time.time() < deadline - 5:
        # Move: Try to add a random set not in family
        # Or remove a random set from family
        # Or swap
        
        # Let's stick to Add/Remove for simplicity first, 
        # but Swap is crucial for improvement.
        
        # Try Swap
        if len(family) > 0:
            # Pick random old
            old_idx = rng.choice(family)
            # Pick random new
            # We want a new that is NOT in family
            # Efficient way: pick random index, if in family, retry or pick from complement
            # Since family is small, just pick random and check
            new_idx = rng.randint(0, total - 1)
            if in_family[new_idx]:
                # Try a few times to find one not in family
                for _ in range(10):
                    new_idx = rng.randint(0, total - 1)
                    if not in_family[new_idx]:
                        break
                else:
                    new_idx = rng.randint(0, total - 1)
                    # If still in family, skip this iteration
                    if in_family[new_idx]:
                        T *= cooling
                        continue

            # Check if swap is valid
            # Current conflict_count[new_idx] includes conflicts with old_idx
            # We need to subtract conflicts caused by (old_idx, A) pairs 
            # where A is in family \ {old_idx} and {old_idx, A, new_idx} is sunflower.
            
            # Actually, conflict_count[new_idx] counts pairs (A,B) in family 
            # such that {A,B,new_idx} is sunflower.
            # If we remove old_idx, the pairs (old_idx, A) are gone.
            # So we need to check for each A in family \ {old_idx} 
            # if {old_idx, A, new_idx} is a sunflower.
            
            valid_swap = True
            for A in family:
                if A == old_idx:
                    continue
                # Check if old_idx, A, new_idx form sunflower
                # i.e., is new_idx in get_conflicts(all_sets[old_idx], all_sets[A])?
                # Instead of calling get_conflicts (which is slowish), 
                # we can just check the condition directly.
                # A_cap_B = all_sets[A] & all_sets[old_idx]
                # A_cap_N = all_sets[A] & all_sets[new_idx]
                # B_cap_N = all_sets[old_idx] & all_sets[new_idx]
                # If A_cap_B == A_cap_N == B_cap_N, then conflict.
                
                cap_old_A = all_sets[old_idx] & all_sets[A]
                cap_old_N = all_sets[old_idx] & all_sets[new_idx]
                cap_A_N = all_sets[A] & all_sets[new_idx]
                
                if cap_old_A == cap_old_N and cap_old_N == cap_A_N:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform Swap
                # 1. Remove old_idx
                family.remove(old_idx)
                in_family[old_idx] = False
                
                # Update conflict counts: decrement for all pairs (old_idx, A) in new family
                for A in family:
                    conflicts = get_conflicts(all_sets[old_idx], all_sets[A])
                    for c in conflicts:
                        conflict_count[c] -= 1
                        
                # 2. Add new_idx
                family.append(new_idx)
                in_family[new_idx] = True
                
                # Update conflict counts: increment for all pairs (new_idx, A) in new family
                for A in family:
                    if A == new_idx:
                        continue
                    conflicts = get_conflicts(all_sets[new_idx], all_sets[A])
                    for c in conflicts:
                        conflict_count[c] += 1
                        
                # Accept swap (size unchanged, but structure changed)
                # In SA, we usually accept if it doesn't increase energy (size drop).
                # Here size is same. We accept to explore state space.
                
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = family[:]
            else:
                # Swap invalid, try simple Add/Remove?
                # Or just skip.
                pass
        else:
            # Family empty, just add
            new_idx = rng.randint(0, total - 1)
            if conflict_count[new_idx] == 0:
                family.append(new_idx)
                in_family[new_idx] = True
                # Update counts (no existing members, so no conflicts to add)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = family[:]

        T *= cooling

    # Final Greedy Polish
    # Try to add any remaining valid sets
    for idx in range(total):
        if time.time() > deadline - 1:
            break
        if not in_family[idx] and conflict_count[idx] == 0:
            family.append(idx)
            in_family[idx] = True
            for j in family[:-1]:
                conflicts = get_conflicts(all_sets[idx], all_sets[j])
                for c in conflicts:
                    conflict_count[c] += 1
            if len(family) > best_size:
                best_size = len(family)
                best_family = family[:]

    # Convert bitmasks back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i+1 for i in range(n) if mask & (1 << i)]
        result.append(elems)
    
    return result
