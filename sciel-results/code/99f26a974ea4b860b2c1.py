import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    
    # 1. Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    
    # 2. Precompute intersection matrix
    # inter[i][j] = all_sets[i] & all_sets[j]
    # Using a list of lists of integers
    inter = [[0] * num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        si = all_sets[i]
        for j in range(num_sets):
            inter[i][j] = si & all_sets[j]
            
    # 3. Helper function to check if set 'idx' can be added to 'family_indices'
    # family_indices is a list of indices into all_sets
    def can_add(idx, family_indices):
        # If family is small, just check pairs
        # Group family members by their intersection with idx
        groups = {}
        for j in family_indices:
            val = inter[idx][j]
            if val not in groups:
                groups[val] = []
            groups[val].append(j)
            
        # For each group, check if any pair (j, k) in the group has inter[j][k] == val
        for val, members in groups.items():
            if len(members) < 2:
                continue
            # Check pairs in members
            m_len = len(members)
            for a in range(m_len):
                ja = members[a]
                for b in range(a + 1, m_len):
                    jb = members[b]
                    if inter[ja][jb] == val:
                        return False
        return True

    best_family = []
    start_time = time.time()
    max_time = 55.0
    
    # Initial greedy construction
    family = []
    in_family = [False] * num_sets
    
    # Sort indices for initial greedy? Random is fine, maybe shuffle once.
    candidates = list(range(num_sets))
    random.shuffle(candidates)
    
    for idx in candidates:
        if time.time() - start_time > max_time:
            break
        if can_add(idx, family):
            family.append(idx)
            in_family[idx] = True
            if len(family) > len(best_family):
                best_family = family[:]
                
    # Local Search / Simulated Annealing style
    # We will try to improve the family by removing one element and adding others.
    
    while time.time() - start_time < max_time:
        current_size = len(family)
        
        # Try to add more sets greedily
        added_any = False
        # Shuffle candidates for diversity
        cand_list = [i for i in range(num_sets) if not in_family[i]]
        random.shuffle(cand_list)
        
        for idx in cand_list:
            if time.time() - start_time > max_time - 1.0:
                break
            if can_add(idx, family):
                family.append(idx)
                in_family[idx] = True
                added_any = True
                if len(family) > len(best_family):
                    best_family = family[:]
                # Restart candidate scan? Or continue?
                # Continuing is faster, but might miss better options.
                # Let's break and restart the outer loop to re-shuffle.
                break
        
        if added_any:
            continue
            
        # If no set could be added, try a swap/remove-add strategy
        if not family:
            break
            
        # Pick a random element to remove
        # We want to remove an element that is "blocking" many others?
        # For now, random removal.
        remove_idx = random.choice(family)
        
        # Save state to revert if necessary
        old_family = family[:]
        old_in_family = in_family[:]
        
        # Remove the element
        family.remove(remove_idx)
        in_family[remove_idx] = False
        
        # Try to add as many sets as possible to the reduced family
        # This might allow adding 1 or more sets.
        # We do a greedy fill on the reduced family.
        
        # Get new candidates
        new_cands = [i for i in range(num_sets) if not in_family[i]]
        random.shuffle(new_cands)
        
        added_count = 0
        for idx in new_cands:
            if time.time() - start_time > max_time - 1.0:
                break
            if can_add(idx, family):
                family.append(idx)
                in_family[idx] = True
                added_count += 1
                # Don't break, try to add more
                # But we need to be careful not to spend too much time here
                if added_count >= 5: # Limit additions per swap attempt
                    break
                    
        # Evaluate
        if len(family) > len(old_family):
            # Improved! Keep it
            if len(family) > len(best_family):
                best_family = family[:]
        else:
            # Revert
            family = old_family
            in_family = old_in_family
            
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
