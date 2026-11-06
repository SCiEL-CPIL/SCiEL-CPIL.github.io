"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Precompute all 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    # Convert to frozensets for faster intersection
    all_frozensets = [frozenset(s) for s in all_sets]
    
    def encode_set(s):
        """Encode a frozenset of elements from 1-16 as an integer bitmask"""
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        return mask
    
    all_masks = [encode_set(s) for s in all_frozensets]
    
    best_family = []
    best_size = 0
    
    # Precompute pairwise intersections for all pairs of sets
    # This is n_total^2 / 2 ≈ 1.66M entries, manageable
    # Store as a 2D list: pair_inter[i][j] = all_masks[i] & all_masks[j]
    # But that's a lot of memory. Instead, compute on the fly but cache frequently used ones.
    
    # Better: use a dictionary to cache intersections
    # Key: (i, j) with i < j, Value: intersection bitmask
    inter_cache = {}
    
    def get_inter(i, j):
        """Get intersection of sets i and j, using cache"""
        if i > j:
            i, j = j, i
        if i == j:
            return all_masks[i]
        key = (i, j)
        if key not in inter_cache:
            inter_cache[key] = all_masks[i] & all_masks[j]
        return inter_cache[key]
    
    # Try multiple random orderings with local search improvement
    max_trials = 30
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_total))
        else:
            order = list(range(n_total))
            random.seed(trial * 7919 + 104729)
            random.shuffle(order)
        
        chosen = []  # list of indices
        chosen_set = set()  # for O(1) membership
        
        # Greedy construction
        for idx in order:
            if time.time() > deadline:
                break
            
            s_mask = all_masks[idx]
            
            # Check if s can be added
            can_add = True
            m = len(chosen)
            
            if m >= 2:
                # Group chosen sets by their intersection with s
                groups = {}
                for i in chosen:
                    inter = get_inter(i, idx)
                    if inter not in groups:
                        groups[inter] = [i]
                    else:
                        groups[inter].append(i)
                
                # For each group, check if any pair has intersection equal to the group key
                for inter, indices in groups.items():
                    if len(indices) >= 2:
                        # Check all pairs in this group
                        n_idx = len(indices)
                        found = False
                        for j in range(n_idx):
                            if found:
                                break
                            for k in range(j + 1, n_idx):
                                a_idx = indices[j]
                                b_idx = indices[k]
                                if get_inter(a_idx, b_idx) == inter:
                                    can_add = False
                                    found = True
                                    break
                    if not can_add:
                        break
            
            if can_add:
                chosen.append(idx)
                chosen_set.add(idx)
        
        # Local search: try to swap or add/remove to improve
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
        
        # Try local improvement on the current family
        if time.time() < deadline - 5.0 and len(chosen) > 0:
            chosen_set = set(chosen)
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Try adding each unchosen set
                for idx in order:
                    if time.time() > deadline - 5.0:
                        break
                    if idx in chosen_set:
                        continue
                    
                    s_mask = all_masks[idx]
                    can_add = True
                    m = len(chosen)
                    
                    if m >= 2:
                        groups = {}
                        for i in chosen:
                            inter = get_inter(i, idx)
                            if inter not in groups:
                                groups[inter] = [i]
                            else:
                                groups[inter].append(i)
                        
                        for inter, indices in groups.items():
                            if len(indices) >= 2:
                                n_idx = len(indices)
                                found = False
                                for j in range(n_idx):
                                    if found:
                                        break
                                    for k in range(j + 1, n_idx):
                                        a_idx = indices[j]
                                        b_idx = indices[k]
                                        if get_inter(a_idx, b_idx) == inter:
                                            can_add = False
                                            found = True
                                            break
                            if not can_add:
                                break
                    
                    if can_add:
                        chosen.append(idx)
                        chosen_set.add(idx)
                        improved = True
                        break
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
    
    # Convert indices back to sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    
    # Verification
    def is_valid_bitmask(family_indices):
        n = len(family_indices)
        masks = [all_masks[i] for i in family_indices]
        for i in range(n):
            for j in range(i + 1, n):
                ab = masks[i] & masks[j]
                for k in range(j + 1, n):
                    ac = masks[i] & masks[k]
                    bc = masks[j] & masks[k]
                    if ab == ac == bc:
                        return False
        return True
    
    if not is_valid_bitmask(best_family):
        # Fallback: rebuild with a simple greedy
        chosen = []
        chosen_set = set()
        for i in range(n_total):
            s_mask = all_masks[i]
            can_add = True
            m = len(chosen)
            if m >= 2:
                groups = {}
                for j in chosen:
                    inter = all_masks[j] & s_mask
                    if inter not in groups:
                        groups[inter] = [j]
                    else:
                        groups[inter].append(j)
                for inter, indices in groups.items():
                    if len(indices) >= 2:
                        n_idx = len(indices)
                        for j in range(n_idx):
                            for k in range(j + 1, n_idx):
                                if (all_masks[indices[j]] & all_masks[indices[k]]) == inter:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                    if not can_add:
                        break
            if can_add:
                chosen.append(i)
                chosen_set.add(i)
        result = [sorted(all_sets[i]) for i in chosen]
    
    return result
# EVOLVE-BLOCK-END
