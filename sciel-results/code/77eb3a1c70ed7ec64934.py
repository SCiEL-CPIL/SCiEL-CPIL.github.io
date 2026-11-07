import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Use k=7 active vertices for more optimization room
    k = 7
    total_masks = 1 << k  # 128 subsets
    
    # Precompute comparability: comp[i][j] = True if i and j are comparable
    # (one is strict subset of the other)
    # For efficiency, precompute for each mask which masks it's comparable to
    comp = [0] * total_masks  # bitmask of comparable sets
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            # Check if comparable: i subset j or j subset i
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)
    
    # Count comparable pairs in a family represented as a bitmask of selected masks
    # For incremental: count how many sets in family are comparable to a given mask
    def count_comparable(mask, family_mask):
        """Count how many sets in family_mask are comparable to mask (excluding mask itself)"""
        # Get all comparable sets to mask
        comp_to_mask = comp[mask]
        # Count how many of those are in family_mask (excluding mask itself)
        intersect = comp_to_mask & family_mask
        # If mask is in family, remove it from count (we don't count self)
        if family_mask & (1 << mask):
            intersect &= ~(1 << mask)
        return bin(intersect).count('1')
    
    def calc_score(family_mask):
        """Calculate total comparable pairs in family"""
        score = 0
        masks_in_family = [i for i in range(total_masks) if family_mask & (1 << i)]
        for idx, i in enumerate(masks_in_family):
            for j in masks_in_family[idx+1:]:
                if comp[i] & (1 << j):
                    score += 1
        return score
    
    def incremental_swap_score(family_mask, remove_mask, add_mask):
        """Calculate score change when removing remove_mask and adding add_mask"""
        # Family without remove_mask
        fam_without = family_mask & ~(1 << remove_mask)
        
        # Edges lost: remove_mask to all in fam_without
        lost = count_comparable(remove_mask, fam_without)
        
        # Edges gained: add_mask to all in fam_without
        gained = count_comparable(add_mask, fam_without)
        
        return gained - lost
    
    # Generate diverse initial families
    def random_initial_family():
        """Generate a random family of m distinct masks"""
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        return all_masks[:m]
    
    def family_to_mask(masks):
        fm = 0
        for m in masks:
            fm |= (1 << m)
        return fm
    
    best_score = -1
    best_family_mask = 0
    
    # Multiple restarts with hill climbing
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialize: mix of strategies
        if num_restarts == 1:
            # Start with small sets (high density of comparability)
            masks_by_size = sorted(range(total_masks), key=lambda x: (bin(x).count('1'), x))
            init_masks = masks_by_size[:m]
        elif num_restarts == 2:
            # Start with a random family
            init_masks = random_initial_family()
        elif num_restarts == 3:
            # Start with medium-sized sets
            masks_by_size = sorted(range(total_masks), key=lambda x: (abs(bin(x).count('1') - k//2), x))
            init_masks = masks_by_size[:m]
        else:
            # Random
            init_masks = random_initial_family()
        
        family_mask = family_to_mask(init_masks)
        current_score = calc_score(family_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_family_mask = family_mask
        
        # Hill climbing with incremental updates
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible swaps
            masks_in_family = [i for i in range(total_masks) if family_mask & (1 << i)]
            masks_not_in_family = [i for i in range(total_masks) if not (family_mask & (1 << i))]
            
            best_delta = 0
            best_remove = -1
            best_add = -1
            
            for rm in masks_in_family:
                for am in masks_not_in_family:
                    delta = incremental_swap_score(family_mask, rm, am)
                    if delta > best_delta:
                        best_delta = delta
                        best_remove = rm
                        best_add = am
            
            if best_delta > 0:
                family_mask = (family_mask & ~(1 << best_remove)) | (1 << best_add)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family_mask = family_mask
        
        # Check if we have time for more restarts
        if time.time() > deadline - 5:
            break
    
    # Convert best family mask to result
    result = []
    for mask in range(total_masks):
        if best_family_mask & (1 << mask):
            verts = [b for b in range(k) if (mask >> b) & 1]
            result.append(verts)
    
    # Ensure exactly m distinct lists
    assert len(result) == m, f"Expected {m} sets, got {len(result)}"
    assert len(set(tuple(x) for x in result)) == m, "Sets are not distinct"
    
    return result
