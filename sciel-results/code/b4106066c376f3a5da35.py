import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    all_masks = 1 << n
    
    def masks_to_sets(masks):
        return [[b for b in range(n) if (mask >> b) & 1] for mask in masks]
    
    def count_comparable(masks):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def generate_candidates(mask, existing_masks, existing_set, count=20):
        """Generate candidate masks that are likely comparable to existing sets."""
        candidates = []
        # Subsets of mask
        sub = mask
        while True:
            if sub != mask and sub not in existing_set:
                candidates.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & mask
        # Supersets of mask
        sup = mask
        while True:
            if sup != mask and sup not in existing_set and sup < all_masks:
                candidates.append(sup)
            if sup == all_masks - 1:
                break
            # Find next superset
            # Add a random bit not in sup
            available = (all_masks - 1) ^ sup
            if available == 0:
                break
            bit = available & -available
            sup = sup | bit
            # Occasionally remove a bit to create different supersets
            if random.random() < 0.3 and sup != mask:
                remove_bits = sup & ~mask
                if remove_bits:
                    rbit = remove_bits & -remove_bits
                    sup = sup ^ rbit
        
        # Also add some random masks for diversity
        for _ in range(count):
            rm = random.randint(0, all_masks - 1)
            if rm not in existing_set:
                candidates.append(rm)
        
        return candidates
    
    def initial_family():
        """Create initial family using a structured approach based on chain-like structure."""
        family = []
        used = set()
        
        # Build a primary chain: {}, {0}, {0,1}, {0,1,2}, ..., {0..11}
        chain_mask = 0
        for k in range(n + 1):
            family.append(chain_mask)
            used.add(chain_mask)
            chain_mask = (1 << (k + 1)) - 1
        
        # We have 13 sets, need 27 more
        # Add sets that are subsets/supersets of chain elements
        # For each chain element, add some variations
        
        # Strategy: For each size, add sets that share many elements with chain
        # Size 1: add {1}, {2}
        for bit in range(1, n):
            mask = 1 << bit
            if mask not in used:
                family.append(mask)
                used.add(mask)
                if len(family) >= m:
                    break
        
        # Size 2: add sets with 0 and another element
        if len(family) < m:
            for bit in range(1, n):
                mask = 1 | (1 << bit)
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    if len(family) >= m:
                        break
        
        # Continue adding sets that are comparable to existing ones
        while len(family) < m:
            # Pick a random existing set and generate a comparable one
            base = random.choice(family)
            # Try subset
            if random.random() < 0.5:
                sub = base
                while sub != 0:
                    if sub not in used:
                        family.append(sub)
                        used.add(sub)
                        break
                    sub = (sub - 1) & base
                else:
                    # Try supersets
                    sup = base
                    while sup < all_masks:
                        if sup not in used:
                            family.append(sup)
                            used.add(sup)
                            break
                        # Add a random bit
                        available = (all_masks - 1) ^ sup
                        if available == 0:
                            break
                        bit = available & -available
                        sup = sup | bit
            else:
                # Try supersets
                sup = base
                while sup < all_masks:
                    if sup not in used:
                        family.append(sup)
                        used.add(sup)
                        break
                    available = (all_masks - 1) ^ sup
                    if available == 0:
                        break
                    bit = available & -available
                    sup = sup | bit
        
        return family[:m]
    
    # Local search: try swapping one set for another to improve count
    def local_search(family, time_limit):
        best = family[:]
        best_score = count_comparable(best)
        
        # Precompute comparability matrix for speed
        def compute_comp_matrix(fam):
            comp = [[False] * m for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    if is_comparable(fam[i], fam[j]):
                        comp[i][j] = True
                        comp[j][i] = True
            return comp
        
        comp = compute_comp_matrix(best)
        existing_set = set(best)
        
        # For each position, try replacing with a different mask
        # Delta = new_comparables - old_comparables
        # When replacing position i with new_mask:
        #   Remove: count of j where comp[i][j] is True
        #   Add: count of j where is_comparable(new_mask, best[j]) and j != i
        
        while time.time() < time_limit:
            improved = False
            # Shuffle order for diversity
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > time_limit:
                    break
                old_comp_count = sum(1 for j in range(m) if j != i and comp[i][j])
                
                # Generate targeted candidates
                candidates = generate_candidates(best[i], best, existing_set, count=15)
                
                # Try candidates in order of promise (those comparable to more existing sets)
                scored_candidates = []
                for cm in candidates:
                    if cm == best[i] or cm in existing_set:
                        continue
                    # Quick score: how many existing sets is it comparable to
                    quick_score = 0
                    for j in range(m):
                        if j != i and is_comparable(cm, best[j]):
                            quick_score += 1
                    if quick_score > old_comp_count:
                        scored_candidates.append((quick_score, cm))
                
                scored_candidates.sort(reverse=True)
                
                # Try top candidates
                for _, new_mask in scored_candidates[:5]:
                    if time.time() > time_limit:
                        break
                    # Full check
                    new_comp_count = 0
                    for j in range(m):
                        if j != i and is_comparable(new_mask, best[j]):
                            new_comp_count += 1
                    
                    delta = new_comp_count - old_comp_count
                    if delta > 0:
                        # Accept the move
                        existing_set.discard(best[i])
                        best[i] = new_mask
                        existing_set.add(new_mask)
                        # Update comp matrix
                        for j in range(m):
                            if j != i:
                                comp[i][j] = comp[j][i] = is_comparable(new_mask, best[j])
                        best_score += delta
                        improved = True
                        break  # Move to next i
        
            if not improved:
                # No improvement found, try perturbation
                # Replace 2-3 random positions with new random masks
                num_replace = random.randint(2, 3)
                replace_indices = random.sample(range(m), num_replace)
                for i in replace_indices:
                    old_mask = best[i]
                    # Generate a new mask that's comparable to some existing sets
                    base = random.choice(best)
                    if random.random() < 0.5:
                        new_mask = base
                        while new_mask != 0:
                            if new_mask not in existing_set or new_mask == old_mask:
                                break
                            new_mask = (new_mask - 1) & base
                        if new_mask == 0 and 0 not in existing_set:
                            new_mask = 0
                        elif new_mask in existing_set and new_mask != old_mask:
                            new_mask = random.randint(0, all_masks - 1)
                    else:
                        new_mask = base
                        while new_mask < all_masks:
                            if new_mask not in existing_set or new_mask == old_mask:
                                break
                            available = (all_masks - 1) ^ new_mask
                            if available == 0:
                                break
                            bit = available & -available
                            new_mask = new_mask | bit
                        if new_mask in existing_set and new_mask != old_mask:
                            new_mask = random.randint(0, all_masks - 1)
                    
                    if new_mask != old_mask and new_mask not in existing_set:
                        existing_set.discard(old_mask)
                        best[i] = new_mask
                        existing_set.add(new_mask)
                        for j in range(m):
                            if j != i:
                                comp[i][j] = comp[j][i] = is_comparable(new_mask, best[j])
        
        return best, best_score
    
    # Run local search with time budget
    family = initial_family()
    family, score = local_search(family, deadline - 3.0)
    
    # Convert to list of lists
    result = masks_to_sets(family)
    
    # Ensure types are correct (Python int, list)
    for i in range(m):
        result[i] = [int(x) for x in result[i]]
    
    return result
