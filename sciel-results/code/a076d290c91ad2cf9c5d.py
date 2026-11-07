"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def calc_score(family):
        """Calculate number of comparable unordered pairs in family."""
        score = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    score += 1
        return score
    
    random.seed(42)
    
    # Generate candidate pools from each size
    by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        by_size[bit_counts[mask]].append(mask)
    
    # Build a good initial solution: thickened chain
    # Start with a chain (one set per size from 0 to 12), then add more sets
    # that are comparable with many existing sets
    
    def build_initial_chain():
        """Build a thickened chain: start with nested sets, add more."""
        family = []
        used = set()
        
        # First, build a chain from size 0 to 12
        # Use masks that are nested: mask_i subset of mask_{i+1}
        # We'll use masks where bits are set in a specific pattern
        
        # Simple approach: use masks 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
        # That's sizes 0,1,2,3,4,5,6,7,8,9,10,11,12 - 13 sets
        chain_masks = [(1 << s) - 1 for s in range(n + 1)]
        
        for mask in chain_masks:
            family.append(mask)
            used.add(mask)
        
        # Now add more sets to reach m=40
        # Add sets that are subsets of the largest set in the chain
        # Or supersets of the smallest
        
        # Strategy: add sets of various sizes that are comparable with many chain members
        # A set of size k is comparable with chain member of size s if:
        # - k <= s and set is subset of chain[s], OR
        # - k >= s and set is superset of chain[s]
        
        # For simplicity, add random sets biased toward small and large sizes
        # Small sets are subsets of many chain members
        # Large sets are supersets of many chain members
        
        # Add small sets (sizes 1-4)
        for size in range(1, 5):
            candidates = [mask for mask in by_size[size] if mask not in used]
            random.shuffle(candidates)
            for mask in candidates:
                if len(family) >= m:
                    break
                family.append(mask)
                used.add(mask)
        
        # Add large sets (sizes 9-12)
        for size in range(9, 13):
            candidates = [mask for mask in by_size[size] if mask not in used]
            random.shuffle(candidates)
            for mask in candidates:
                if len(family) >= m:
                    break
                family.append(mask)
                used.add(mask)
        
        # If still not enough, add random sets
        if len(family) < m:
            all_candidates = [mask for mask in range(total_masks) if mask not in used]
            random.shuffle(all_candidates)
            for mask in all_candidates:
                if len(family) >= m:
                    break
                family.append(mask)
                used.add(mask)
        
        return family[:m]
    
    family = build_initial_chain()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score
    
    # Local search with multiple restarts
    no_improve_count = 0
    max_no_improve = 2000
    restarts = 0
    max_restarts = 3
    
    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Try a restart with a different random seed
            restarts += 1
            if restarts > max_restarts:
                break
            
            # New random initialization
            random.seed(42 + restarts * 1000)
            family = build_initial_chain()
            current_score = calc_score(family)
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
            no_improve_count = 0
            continue
        
        # Try random swaps
        idx = random.randint(0, m - 1)
        old_mask = family[idx]
        
        # Pick a new mask from a random size, biased toward small and large
        r = random.random()
        if r < 0.4:
            new_size = random.randint(0, 4)
        elif r < 0.8:
            new_size = random.randint(8, 12)
        else:
            new_size = random.randint(0, n)
        
        if not by_size[new_size]:
            continue
        new_mask = random.choice(by_size[new_size])
        if new_mask == old_mask:
            continue
        # Ensure distinctness
        if new_mask in family:
            continue
        
        # Calculate delta score
        delta = 0
        for j in range(m):
            if j == idx:
                continue
            other = family[j]
            if is_comparable(old_mask, other):
                delta -= 1
            if is_comparable(new_mask, other):
                delta += 1
        
        if delta > 0:
            family[idx] = new_mask
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1
    
    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
