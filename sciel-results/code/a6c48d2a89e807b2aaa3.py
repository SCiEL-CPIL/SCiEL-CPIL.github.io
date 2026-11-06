import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    num_masks = len(all_masks)
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    # Precompute popcount for masks up to 2^16? Or just use bit_count() (Python 3.10+)
    # To be safe, use bin(x).count('1') or a lookup table.
    # Lookup table for 16-bit integers
    popcount = np.zeros(1 << n, dtype=np.int8)
    for i in range(1 << n):
        popcount[i] = bin(i).count('1')
        
    # conflict_count[i] is the number of pairs (A, B) in current family 
    # such that mask i would form a sunflower with A and B.
    # i.e., A & B == A & mask_i == B & mask_i
    conflict_count = np.zeros(num_masks, dtype=np.int32)
    
    family = []
    in_family = np.zeros(num_masks, dtype=bool)
    
    def add_to_family(mask_idx):
        """Add mask to family and update conflict counts."""
        mask = all_masks[mask_idx]
        family.append(mask_idx)
        in_family[mask_idx] = True
        
        # Update conflict counts for all existing masks in family
        # For each Y in family (excluding the new one), 
        # find all C such that (mask & Y) == (mask & C) == (Y & C)
        # Let K = mask & Y
        # Let D = (mask | Y) & ~K  # elements in mask or Y but not in intersection
        # C must contain K, be disjoint from D, and have size 4.
        # C = K | T, where T is subset of ~(K | D) of size 4 - popcount(K)
        
        for other_idx in family[:-1]:
            other_mask = all_masks[other_idx]
            K = mask & other_mask
            D = (mask | other_mask) & ~K
            avail = ~(K | D) & ((1 << n) - 1)
            
            k_size = popcount[K]
            t_size = k - k_size
            
            if t_size < 0 or t_size > n:
                continue
                
            # Find all subsets T of avail of size t_size
            # avail has popcount 16 - |K| - |D|
            # Since K and D are disjoint, |K| + |D| = |mask U Y| = 8 - k_size
            # So avail size is 16 - (8 - k_size) = 8 + k_size
            # We need to choose t_size = 4 - k_size from 8 + k_size.
            
            # Get list of bits in avail
            avail_bits = [i for i in range(n) if avail & (1 << i)]
            
            for T_combo in combinations(avail_bits, t_size):
                T = 0
                for b in T_combo:
                    T |= (1 << b)
                C = K | T
                c_idx = mask_to_idx.get(C)
                if c_idx is not None:
                    conflict_count[c_idx] += 1

    def remove_from_family(mask_idx):
        """Remove mask from family and update conflict counts."""
        mask = all_masks[mask_idx]
        family.remove(mask_idx)
        in_family[mask_idx] = False
        
        # Decrement conflict counts
        for other_idx in family:
            other_mask = all_masks[other_idx]
            K = mask & other_mask
            D = (mask | other_mask) & ~K
            avail = ~(K | D) & ((1 << n) - 1)
            
            k_size = popcount[K]
            t_size = k - k_size
            
            if t_size < 0 or t_size > n:
                continue
                
            avail_bits = [i for i in range(n) if avail & (1 << i)]
            
            for T_combo in combinations(avail_bits, t_size):
                T = 0
                for b in T_combo:
                    T |= (1 << b)
                C = K | T
                c_idx = mask_to_idx.get(C)
                if c_idx is not None:
                    conflict_count[c_idx] -= 1

    best_size = 0
    best_family = []
    
    # Random restarts
    for trial in range(100):
        if time.time() > deadline - 20:
            break
            
        random.shuffle(all_masks)
        family = []
        in_family = np.zeros(num_masks, dtype=bool)
        conflict_count = np.zeros(num_masks, dtype=np.int32)
        
        # Greedy Phase
        for mask_idx in range(num_masks):
            if conflict_count[mask_idx] == 0:
                # Add it
                mask = all_masks[mask_idx]
                family.append(mask_idx)
                in_family[mask_idx] = True
                
                for other_idx in family[:-1]:
                    other_mask = all_masks[other_idx]
                    K = mask & other_mask
                    D = (mask | other_mask) & ~K
                    avail = ~(K | D) & ((1 << n) - 1)
                    
                    k_size = popcount[K]
                    t_size = k - k_size
                    
                    if t_size < 0 or t_size > n:
                        continue
                    
                    avail_bits = [i for i in range(n) if avail & (1 << i)]
                    
                    for T_combo in combinations(avail_bits, t_size):
                        T = 0
                        for b in T_combo:
                            T |= (1 << b)
                        C = K | T
                        c_idx = mask_to_idx.get(C)
                        if c_idx is not None:
                            conflict_count[c_idx] += 1
                            
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Local Search Phase
        # Try swaps: remove i, add j
        # If conflict_count[j] == 0 after removing i, swap is valid.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle family order for randomization in search
            # We iterate over all pairs (i in family, j not in family)
            # To speed up, we can sample.
            
            # Create list of indices not in family
            # This is expensive if done every time. 
            # But num_masks is 1820.
            
            # We can just iterate j from 0 to num_masks-1
            # and check if not in_family[j]
            
            for i_idx in family[:]: # Iterate over copy
                if time.time() > deadline - 5:
                    break
                
                # Try to remove i_idx
                mask_i = all_masks[i_idx]
                family.remove(i_idx)
                in_family[i_idx] = False
                
                # Update conflict counts (decrement)
                for other_idx in family:
                    other_mask = all_masks[other_idx]
                    K = mask_i & other_mask
                    D = (mask_i | other_mask) & ~K
                    avail = ~(K | D) & ((1 << n) - 1)
                    
                    k_size = popcount[K]
                    t_size = k - k_size
                    
                    if t_size < 0 or t_size > n:
                        continue
                    
                    avail_bits = [i for i in range(n) if avail & (1 << i)]
                    
                    for T_combo in combinations(avail_bits, t_size):
                        T = 0
                        for b in T_combo:
                            T |= (1 << b)
                        C = K | T
                        c_idx = mask_to_idx.get(C)
                        if c_idx is not None:
                            conflict_count[c_idx] -= 1
                            
                # Now try to find a j not in family with conflict_count[j] == 0
                # Randomize j selection
                # We can pick a random few j's
                for _ in range(10): # Try 10 random candidates
                    j_idx = random.randint(0, num_masks - 1)
                    if not in_family[j_idx] and conflict_count[j_idx] == 0:
                        # Accept swap
                        mask_j = all_masks[j_idx]
                        family.append(j_idx)
                        in_family[j_idx] = True
                        
                        # Update conflict counts (increment) for j
                        for other_idx in family[:-1]:
                            other_mask = all_masks[other_idx]
                            K = mask_j & other_mask
                            D = (mask_j | other_mask) & ~K
                            avail = ~(K | D) & ((1 << n) - 1)
                            
                            k_size = popcount[K]
                            t_size = k - k_size
                            
                            if t_size < 0 or t_size > n:
                                continue
                            
                            avail_bits = [i for i in range(n) if avail & (1 << i)]
                            
                            for T_combo in combinations(avail_bits, t_size):
                                T = 0
                                for b in T_combo:
                                    T |= (1 << b)
                                C = K | T
                                c_idx = mask_to_idx.get(C)
                                if c_idx is not None:
                                    conflict_count[c_idx] += 1
                            
                        improved = True
                        break # Break inner loop, continue outer
                            
                # If no swap found, revert removal of i_idx
                if not improved or in_family[i_idx]: 
                     # This logic is tricky. If we broke out of j loop due to success, in_family[i_idx] is still False.
                     # We need to know if we added a j.
                     pass 
                
                # Simpler logic:
                # We removed i. 
                # If we found a j and added it, we keep the state.
                # If we didn't find a j, we must re-add i.
                
                # Let's refactor the swap logic to be cleaner.
                
        # The above loop structure is messy. Let's simplify.
        
    # The messy local search above might have bugs. Let's use a cleaner one.
    
    # Re-run with a cleaner local search on the best family found so far
    
    # Reconstruct state for best_family
    # This is O(N^2 * K) which is fine for N=15.
    
    final_family = best_family[:]
    in_family = np.zeros(num_masks, dtype=bool)
    for idx in final_family:
        in_family[idx] = True
        
    conflict_count = np.zeros(num_masks, dtype=np.int32)
    for i in range(len(final_family)):
        for j in range(i+1, len(final_family)):
            mask_i = all_masks[final_family[i]]
            mask_j = all_masks[final_family[j]]
            K = mask_i & mask_j
            D = (mask_i | mask_j) & ~K
            avail = ~(K | D) & ((1 << n) - 1)
            
            k_size = popcount[K]
            t_size = k - k_size
            
            if t_size < 0 or t_size > n:
                continue
            
            avail_bits = [i for i in range(n) if avail & (1 << i)]
            
            for T_combo in combinations(avail_bits, t_size):
                T = 0
                for b in T_combo:
                    T |= (1 << b)
                C = K | T
                c_idx = mask_to_idx.get(C)
                if c_idx is not None:
                    conflict_count[c_idx] += 1
                    # Wait, conflict_count counts pairs. 
                    # Each pair (i,j) contributes 1 to each valid C.
                    # So this is correct.

    # Local Search: Swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Randomly shuffle family to try different removals
        # We iterate through family indices
        for i_idx in final_family[:]:
            if time.time() > deadline - 5:
                break
                
            # Remove i_idx
            mask_i = all_masks[i_idx]
            final_family.remove(i_idx)
            in_family[i_idx] = False
            
            # Decrement conflicts for pairs involving i_idx
            for other_idx in final_family:
                other_mask = all_masks[other_idx]
                K = mask_i & other_mask
                D = (mask_i | other_mask) & ~K
                avail = ~(K | D) & ((1 << n) - 1)
                
                k_size = popcount[K]
                t_size = k - k_size
                
                if t_size < 0 or t_size > n:
                    continue
                
                avail_bits = [i for i in range(n) if avail & (1 << i)]
                
                for T_combo in combinations(avail_bits, t_size):
                    T = 0
                    for b in T_combo:
                        T |= (1 << b)
                    C = K | T
                    c_idx = mask_to_idx.get(C)
                    if c_idx is not None:
                        conflict_count[c_idx] -= 1
                        
            # Try to add a random j
            # Sample 50 candidates
            swapped = False
            for _ in range(50):
                j_idx = random.randint(0, num_masks - 1)
                if not in_family[j_idx] and conflict_count[j_idx] == 0:
                    # Add j_idx
                    mask_j = all_masks[j_idx]
                    final_family.append(j_idx)
                    in_family[j_idx] = True
                    
                    # Increment conflicts for pairs involving j_idx
                    for other_idx in final_family[:-1]:
                        other_mask = all_masks[other_idx]
                        K = mask_j & other_mask
                        D = (mask_j | other_mask) & ~K
                        avail = ~(K | D) & ((1 << n) - 1)
                        
                        k_size = popcount[K]
                        t_size = k - k_size
                        
                        if t_size < 0 or t_size > n:
                            continue
                        
                        avail_bits = [i for i in range(n) if avail & (1 << i)]
                        
                        for T_combo in combinations(avail_bits, t_size):
                            T = 0
                            for b in T_combo:
                                T |= (1 << b)
                            C = K | T
                            c_idx = mask_to_idx.get(C)
                            if c_idx is not None:
                                conflict_count[c_idx] += 1
                                
                    improved = True
                    swapped = True
                    break
            
            if not swapped:
                # Revert: Re-add i_idx
                mask_i = all_masks[i_idx]
                final_family.append(i_idx)
                in_family[i_idx] = True
                
                # Increment conflicts for pairs involving i_idx
                for other_idx in final_family[:-1]:
                    other_mask = all_masks[other_idx]
                    K = mask_i & other_mask
                    D = (mask_i | other_mask) & ~K
                    avail = ~(K | D) & ((1 << n) - 1)
                    
                    k_size = popcount[K]
                    t_size = k - k_size
                    
                    if t_size < 0 or t_size > n:
                        continue
                    
                    avail_bits = [i for i in range(n) if avail & (1 << i)]
                    
                    for T_combo in combinations(avail_bits, t_size):
                        T = 0
                        for b in T_combo:
                            T |= (1 << b)
                        C = K | T
                        c_idx = mask_to_idx.get(C)
                        if c_idx is not None:
                            conflict_count[c_idx] += 1

    # Convert to lists
    result = []
    for idx in final_family:
        mask = all_masks[idx]
        elements = [i+1 for i in range(n) if mask & (1 << i)]
        result.append(elements)
        
    return result

