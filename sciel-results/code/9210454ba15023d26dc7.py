# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin
    
    # Precompute all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    mask_to_idx = {}
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
    
    N = len(all_masks)
    ALL_MASK = (1 << 16) - 1
    
    # Precompute popcount for 16-bit integers? Or just use bin(x).count('1')
    # Since 16 bits, we can use a lookup table or just the builtin.
    
    def get_conflict_masks(a, b):
        """
        Returns a list of masks c such that a, b, c form a sunflower.
        Condition: a&b == a&c == b&c
        """
        i = a & b
        i_size = bin(i).count('1')
        # If i_size == 4, a==b, but we assume distinct
        # z is the set of elements not in a or b
        z = ALL_MASK ^ (a | b)
        
        k = 4 - i_size
        if k < 0:
            return []
        
        # Get indices of set bits in z
        z_indices = [j for j in range(16) if (z >> j) & 1]
        
        conflicts = []
        for combo in combinations(z_indices, k):
            c_mask = i
            for idx in combo:
                c_mask |= (1 << idx)
            # c_mask is a valid 4-set
            conflicts.append(c_mask)
        return conflicts

    def greedy_with_local_search(seed):
        random.seed(seed)
        
        # counts[i] is the number of pairs in F that conflict with all_masks[i]
        counts = [0] * N
        in_family = [False] * N
        family_indices = []
        available = set(range(N))
        
        # Helper to update counts when adding a mask index
        def add_to_family(idx):
            in_family[idx] = True
            family_indices.append(idx)
            available.discard(idx)
            m_idx = all_masks[idx]
            for other_idx in family_indices:
                if other_idx == idx:
                    continue
                m_other = all_masks[other_idx]
                conflicts = get_conflict_masks(m_idx, m_other)
                for c_mask in conflicts:
                    c_idx = mask_to_idx[c_mask]
                    counts[c_idx] += 1

        # Helper to update counts when removing a mask index
        def remove_from_family(idx):
            in_family[idx] = False
            family_indices.remove(idx)
            available.add(idx)
            m_idx = all_masks[idx]
            for other_idx in family_indices:
                m_other = all_masks[other_idx]
                conflicts = get_conflict_masks(m_idx, m_other)
                for c_mask in conflicts:
                    c_idx = mask_to_idx[c_mask]
                    counts[c_idx] -= 1

        while available:
            if time.time() > deadline:
                break
            
            # Find candidates with 0 conflicts
            zero_conflict = [c for c in available if counts[c] == 0]
            
            if zero_conflict:
                # Pick randomly from zero conflicts to diversify
                idx = random.choice(zero_conflict)
                add_to_family(idx)
            else:
                # Local search: try to swap
                swapped = False
                # Try removing each element in family
                # To make it faster, shuffle the order we try
                fam_copy = family_indices[:]
                random.shuffle(fam_copy)
                
                for a_idx in fam_copy:
                    if time.time() > deadline:
                        break
                    
                    # Remove a_idx
                    remove_from_family(a_idx)
                    
                    # Check if any available now has 0 conflicts
                    # Note: counts for available sets have been updated (decremented)
                    new_zero = [c for c in available if counts[c] == 0]
                    
                    if new_zero:
                        # Pick one
                        c_idx = random.choice(new_zero)
                        # Add c_idx
                        add_to_family(c_idx)
                        swapped = True
                        break
                    else:
                        # Re-add a_idx
                        add_to_family(a_idx)
                
                if not swapped:
                    break
        
        return family_indices

    best_family = []
    
    # Run multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        family = greedy_with_local_search(seed)
        if len(family) > len(best_family):
            best_family = family
            # Early stop if we hit the theoretical max or a high threshold?
            # Max is 250.
            if len(best_family) >= 250:
                break

    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
