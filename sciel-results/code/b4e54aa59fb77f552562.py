import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    candidates = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        
    # Helper to check if s can be added to family
    def can_add(s, family):
        m = len(family)
        for i in range(m):
            a = family[i]
            for j in range(i + 1, m):
                b = family[j]
                if (a & b) == (a & s) == (b & s):
                    return False
        return True

    best_family = []
    best_size = 0
    start_time = time.time()
    time_limit = 58.0 # 2s margin for 60s limit

    # Initial greedy pass
    random.shuffle(candidates)
    family = []
    for s in candidates:
        if can_add(s, family):
            family.append(s)
            
    best_family = family[:]
    best_size = len(family)

    # Local Search: Remove one, try to fill
    # We keep a list of candidates not in family
    # Actually, just iterate all candidates in each restart
    
    while time.time() - start_time < time_limit:
        # Pick a random set from current family to remove
        if not family:
            break
            
        idx_to_remove = random.randint(0, len(family) - 1)
        removed_set = family[idx_to_remove]
        
        # Create a new family without the removed set
        new_family = []
        for i, s in enumerate(family):
            if i != idx_to_remove:
                new_family.append(s)
                
        # Try to add candidates
        # Shuffle candidates to randomize
        random.shuffle(candidates)
        
        # Optimization: We can skip candidates that are in new_family
        # But checking membership in a list is O(M). 
        # Better to use a set of masks for new_family for O(1) check?
        # M is small (~100). Set creation is O(M).
        new_family_set = set(new_family)
        
        for s in candidates:
            if s in new_family_set:
                continue
            if can_add(s, new_family):
                new_family.append(s)
                new_family_set.add(s)
                
        if len(new_family) > len(family):
            family = new_family
            best_family = family[:]
            best_size = len(family)
            
            if best_size >= 250:
                break
        # If not improved, we keep the old family and try again with a different removal
        # We don't revert family, we just tried a move. If it didn't work, we try another move.
        # Wait, the logic above: 
        # 1. Take current family.
        # 2. Remove one.
        # 3. Greedy fill.
        # 4. If better, update family.
        # 5. If not, family remains as it was before the removal attempt.
        # This is correct.

    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
