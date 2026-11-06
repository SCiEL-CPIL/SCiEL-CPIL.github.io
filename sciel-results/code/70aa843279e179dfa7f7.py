import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    k = 4
    
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    
    def is_compatible(c_mask, family):
        L = len(family)
        for i in range(L):
            a = family[i]
            for j in range(i + 1, L):
                b = family[j]
                ab = a & b
                ac = a & c_mask
                bc = b & c_mask
                if ab == ac == bc:
                    return False
        return True

    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        current_family = []
        indices = list(range(num_sets))
        random.shuffle(indices)
        
        # Greedy Phase
        for idx in indices:
            if time.time() > deadline:
                break
            c_mask = all_sets[idx]
            if is_compatible(c_mask, current_family):
                current_family.append(c_mask)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family.copy()
            
            # Local Search: Remove 1, Re-Greedy
            # We try to see if removing one element allows us to pack more.
            # We do this a few times.
            local_improved = True
            # Limit local search to 1s to allow more restarts
            local_deadline = time.time() + 1.0
            
            while local_improved and time.time() < local_deadline:
                local_improved = False
                if not best_family:
                    break
                
                # Pick random element to remove
                rem_idx = random.randint(0, len(best_family) - 1)
                rem_mask = best_family[rem_idx]
                
                # Create base family without rem_mask
                base_family = best_family[:rem_idx] + best_family[rem_idx+1:]
                
                # Run greedy to fill
                # We need to consider all sets NOT in base_family
                # To avoid checking sets already in base_family, we can just check compatibility.
                # If a set is in base_family, it is compatible with itself? No, distinct sets.
                # But we shouldn't add duplicates.
                # So we maintain a set of masks in the family.
                
                family_set = set(base_family)
                new_family = base_family.copy()
                
                # Shuffle all sets again
                idxs = list(range(num_sets))
                random.shuffle(idxs)
                
                for idx in idxs:
                    if time.time() > local_deadline:
                        break
                    c_mask = all_sets[idx]
                    if c_mask in family_set:
                        continue
                    if is_compatible(c_mask, new_family):
                        new_family.append(c_mask)
                        family_set.add(c_mask)
                        
                if len(new_family) > len(best_family):
                    best_family = new_family
                    best_size = len(best_family)
                    local_improved = True

    result = []
    for mask in best_family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(s)
        
    return result
