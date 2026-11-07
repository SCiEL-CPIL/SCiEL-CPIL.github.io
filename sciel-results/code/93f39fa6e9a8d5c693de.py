"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute bit counts
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute comparability masks
    # subsets_mask[C] has bits set for all S such that S < C (S is strict subset of C)
    # supersets_mask[C] has bits set for all S such that C < S (C is strict subset of S)
    subsets_mask = [0] * total_masks
    supersets_mask = [0] * total_masks
    
    # This O(N^2) precomputation is 4096^2 ~ 16M ops, which is fast in Python if optimized?
    # 16M bitwise ops might take a second or two in pure Python. 
    # We can optimize by iterating by size.
    
    # Better: 
    # For each mask C, subsets are all masks S where S & ~C == 0 and S != C.
    # We can generate them efficiently.
    
    # Actually, let's just do the O(N^2) but with early breaks or simpler logic if possible.
    # Or use NumPy? No, Python ints are arbitrary precision and fast for bitwise.
    # 16M iterations in Python might be ~1-2 seconds. Acceptable.
    
    for c in range(total_masks):
        # Find all strict subsets
        sub = c
        # Iterate through all submasks of c? 
        # Standard trick:
        s = c
        while True:
            if s != c:
                subsets_mask[c] |= (1 << s)
            if s == 0:
                break
            s = (s - 1) & c
            
        # Find all strict supersets
        # Supersets of c are masks S where S | c == S and S != c.
        # Equivalent to: S = c | X where X is a subset of ~c.
        # ~c in n bits is ((1<<n)-1) ^ c.
        available = ((1 << n) - 1) ^ c
        x = available
        while True:
            sup = c | x
            if sup != c:
                supersets_mask[c] |= (1 << sup)
            if x == 0:
                break
            x = (x - 1) & available

    # Helper to count bits in a Python int
    def popcount(x):
        return bin(x).count('1')

    def calc_score(family_list):
        # family_list is a list of masks
        # Score is number of comparable pairs
        # We can use the precomputed masks
        # But for initial check, simple loop is fine
        score = 0
        for i in range(len(family_list)):
            for j in range(i + 1, len(family_list)):
                if (family_list[i] & family_list[j]) == family_list[i] or (family_list[i] & family_list[j]) == family_list[j]:
                    score += 1
        return score

    def greedy_initial():
        family = []
        used = set()
        # Start with a "spine" to ensure good connectivity
        # Spine: 0, 1, 3, 7, ... (powers of 2 - 1)
        # This gives a chain of length 13 (sizes 0 to 12)
        spine = [ (1 << (1 << i)) - 1 for i in range(n) ] # This is wrong, 1<<i is size
        # Correct spine:
        spine = []
        mask = 0
        for i in range(n):
            spine.append(mask)
            mask |= (1 << i)
        spine.append(mask) # Add the full set
        # spine has 13 elements: size 0 to 12
        
        for s in spine:
            if s not in used:
                family.append(s)
                used.add(s)
        
        # Fill remaining
        while len(family) < m:
            best_mask = None
            best_comparables = -1
            # Randomly sample candidates
            for _ in range(50):
                cand = random.randint(0, total_masks - 1)
                if cand in used:
                    continue
                # Count comparables
                comps = popcount((subsets_mask[cand] | supersets_mask[cand]) & 
                                 sum([1 << f for f in family]))
                if comps > best_comparables:
                    best_comparables = comps
                    best_mask = cand
            if best_mask is None:
                # Fallback
                for cand in range(total_masks):
                    if cand not in used:
                        best_mask = cand
                        break
            family.append(best_mask)
            used.add(best_mask)
            
        return family

    # State representation:
    # family: list of masks
    # family_mask: int with bits set for masks in family
    
    family = greedy_initial()
    family_mask = 0
    for f in family:
        family_mask |= (1 << f)
    
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score
    best_family_mask = family_mask

    # Local Search with Simulated Annealing
    temp = 10.0
    cooling = 0.999
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute a list of all masks for random selection
    all_masks = list(range(total_masks))

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random index to replace
        idx = random.randint(0, m - 1)
        old_mask = family[idx]
        
        # Generate a candidate
        # Bias towards masks that are comparable with many existing
        # Simple: random mask
        new_mask = random.choice(all_masks)
        
        if new_mask == old_mask:
            continue
            
        # Check if new_mask is already in family (and not at idx)
        if (family_mask >> new_mask) & 1:
            # If it's at idx, it's the same. If elsewhere, it's a duplicate.
            # Since we replace, if new_mask is in family at j != idx, we create duplicate.
            # We must avoid duplicates.
            # If new_mask is in family, we can't add it unless we remove the other one?
            # The move is "replace idx with new_mask".
            # If new_mask is already in family at j != idx, the resulting family has a duplicate.
            # We should reject or handle. Simplest: reject.
            continue

        # Calculate delta
        # old_gain: number of comparables of old_mask in family (excluding itself)
        # The mask `family_mask` includes old_mask.
        # subsets_mask[old_mask] and supersets_mask[old_mask] do NOT include old_mask.
        # So `popcount(family_mask & (subsets_mask[old_mask] | supersets_mask[old_mask]))` is correct.
        old_gain = popcount(family_mask & (subsets_mask[old_mask] | supersets_mask[old_mask]))
        
        # Remove old_mask from family_mask to calculate new_gain
        temp_family_mask = family_mask & ~(1 << old_mask)
        
        # new_gain: number of comparables of new_mask in (family \ {old_mask})
        new_gain = popcount(temp_family_mask & (subsets_mask[new_mask] | supersets_mask[new_mask]))
        
        delta = new_gain - old_gain
        
        if delta > 0 or (random.random() < math.exp(delta / temp)):
            # Accept
            family[idx] = new_mask
            family_mask = temp_family_mask | (1 << new_mask)
            current_score += delta
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
                best_family_mask = family_mask
        else:
            no_improve += 1
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01 # Floor

    # Final polish: Exhaustive single-swap improvement
    # This is O(M * N) where N is 4096. 40 * 4096 = 163k ops. Very fast.
    final_family = list(best_family)
    final_mask = best_family_mask
    final_score = best_score
    
    for _ in range(10): # Multiple passes
        improved = False
        for idx in range(m):
            old_mask = final_family[idx]
            best_delta = 0
            best_new = None
            
            # Try all other masks
            # To speed up, we can just iterate all 4096
            for new_mask in range(total_masks):
                if new_mask == old_mask:
                    continue
                if (final_mask >> new_mask) & 1:
                    continue
                    
                old_gain = popcount(final_mask & (subsets_mask[old_mask] | supersets_mask[old_mask]))
                temp_fm = final_mask & ~(1 << old_mask)
                new_gain = popcount(temp_fm & (subsets_mask[new_mask] | supersets_mask[new_mask]))
                delta = new_gain - old_gain
                
                if delta > best_delta:
                    best_delta = delta
                    best_new = new_mask
            
            if best_new is not None and best_delta > 0:
                final_family[idx] = best_new
                final_mask = (final_mask & ~(1 << old_mask)) | (1 << best_new)
                final_score += best_delta
                improved = True
        
        if not improved:
            break

    if final_score > best_score:
        best_score = final_score
        best_family = final_family

    # Convert to lists
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    return result
# EVOLVE-BLOCK-END
