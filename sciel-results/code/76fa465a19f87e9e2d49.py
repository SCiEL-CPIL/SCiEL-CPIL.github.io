import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n+1), k))
    # Convert to bitmasks for fast intersection
    all_masks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << x)
        all_masks.append(mask)
    
    # Map mask to original set for output
    mask_to_set = {m: s for m, s in zip(all_masks, all_sets)}
    
    # Precompute pairwise intersections? No, just compute on the fly.
    # But we can precompute intersection of every pair of masks in all_masks?
    # 1820 * 1820 = 3.3M pairs. Storing a dict or array might be slow/memory heavy.
    # On-the-fly is fine.
    
    def can_add(candidate, family):
        """
        Check if candidate can be added to family without creating a sunflower.
        A sunflower is three distinct sets A, B, C with A&B = B&C = C&A.
        If we add C to {A, B}, we check if C&A == A&B and C&B == A&B.
        """
        # family is a list of masks
        # We need to check all pairs in family
        len_fam = len(family)
        for i in range(len_fam):
            a = family[i]
            for j in range(i + 1, len_fam):
                b = family[j]
                s_ab = a & b
                # Check if candidate forms a sunflower with a and b
                if (candidate & a == s_ab) and (candidate & b == s_ab):
                    return False
        return True

    def greedy_construct(order):
        fam = []
        for m in order:
            if can_add(m, fam):
                fam.append(m)
        return fam

    best_size = 0
    best_fam = []
    
    # Multi-start with random orders
    while time.time() < deadline:
        # Random shuffle
        random.shuffle(all_masks)
        
        # Greedy phase
        current_fam = greedy_construct(all_masks)
        
        # Local Search / Hill Climbing
        # Try to improve by swapping or adding
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any set not in family
            fam_set = set(current_fam)
            candidates = [m for m in all_masks if m not in fam_set]
            random.shuffle(candidates)
            
            added = False
            for c in candidates:
                if can_add(c, current_fam):
                    current_fam.append(c)
                    fam_set.add(c)
                    added = True
                    improved = True
                    break
            
            # 2. If no single addition, try a swap: remove one, add one
            if not added:
                # Pick a random set to remove
                if current_fam:
                    idx = random.randint(0, len(current_fam) - 1)
                    removed = current_fam[idx]
                    current_fam.pop(idx)
                    fam_set.discard(removed)
                    
                    # Try to add a better one
                    success = False
                    for c in candidates: # candidates list is stale, regenerate or use all_masks
                         # Using all_masks is safer
                         pass
                     
                    # Let's just try all masks not in current family
                    for c in all_masks:
                        if c in fam_set:
                            continue
                        if can_add(c, current_fam):
                            current_fam.append(c)
                            fam_set.add(c)
                            success = True
                            improved = True
                            break
                    
                    if not success:
                        # Put it back
                        current_fam.append(removed)
                        fam_set.add(removed)
            
            # 3. Try 1-for-2 swap? (Remove 1, add 2)
            # This is complex to implement efficiently. 
            # Let's stick to 1-for-1 and 0-for-1 for now.
            # If we get stuck, restart.
            
            # Safety break if close to deadline
            if time.time() > deadline - 2.0:
                break

        if len(current_fam) > best_size:
            best_size = len(current_fam)
            best_fam = current_fam[:]
            
        # If we found a very large family, we might be close to max.
        # Max possible is 1820. 
        # Known bounds for sunflower-free families:
        # For k=4, n=16.
        # Let's just run until deadline.

    # Convert best_fam masks back to sorted lists
    return [sorted(mask_to_set[m]) for m in best_fam]

# EVOLVE-BLOCK-START
# The code above is the solution.
# EVOLVE-BLOCK-END
