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
    
    # Precompute comparability matrix: comp[a][b] = 1 if a and b are comparable
    # This is a 4096x4096 matrix = 16M entries, stored as a flat list of bytes
    comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if (a & b) == a or (a & b) == b:
                comp[base + b] = 1
    
    def is_comparable(a, b):
        return comp[a * total_masks + b] == 1
    
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
    
    # Build a good initial solution using greedy approach
    def build_greedy_initial():
        """Build initial solution by greedily adding most comparable sets."""
        family = []
        used = set()
        
        # Start with a chain to ensure a good base
        chain_masks = [(1 << s) - 1 for s in range(n + 1)]
        for mask in chain_masks:
            family.append(mask)
            used.add(mask)
        
        # Now greedily add sets that maximize comparability with existing family
        remaining_slots = m - len(family)
        for _ in range(remaining_slots):
            best_mask = None
            best_comp_count = -1
            # Try masks from small and large sizes first
            for size in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 5, 6, 7]:
                candidates = [mask for mask in by_size[size] if mask not in used]
                random.shuffle(candidates)
                # Only check first few candidates for speed
                for mask in candidates[:50]:
                    comp_count = sum(1 for f in family if is_comparable(mask, f))
                    if comp_count > best_comp_count:
                        best_comp_count = comp_count
                        best_mask = mask
                if best_mask is not None and best_comp_count >= len(family) * 0.7:
                    break
            
            if best_mask is not None:
                family.append(best_mask)
                used.add(best_mask)
            else:
                # Fallback: add any unused mask
                all_remaining = [mask for mask in range(total_masks) if mask not in used]
                if all_remaining:
                    random.shuffle(all_remaining)
                    family.append(all_remaining[0])
                    used.add(all_remaining[0])
        
        return family[:m]
    
    family = build_greedy_initial()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score
    
    # Local search with improved strategies
    no_improve_count = 0
    max_no_improve = 5000
    restarts = 0
    max_restarts = 5
    
    # Precompute family set for O(1) membership check
    family_set = set(family)
    
    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Try a restart with a different random seed
            restarts += 1
            if restarts > max_restarts:
                break
            
            # New random initialization
            random.seed(42 + restarts * 1000)
            family = build_greedy_initial()
            family_set = set(family)
            current_score = calc_score(family)
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
            no_improve_count = 0
            continue
        
        # Try random swaps with bias toward promising sizes
        idx = random.randint(0, m - 1)
        old_mask = family[idx]
        
        # Pick a new mask with bias toward small and large sizes
        r = random.random()
        if r < 0.35:
            new_size = random.randint(0, 3)
        elif r < 0.70:
            new_size = random.randint(9, 12)
        elif r < 0.85:
            new_size = random.randint(4, 8)
        else:
            new_size = random.randint(0, n)
        
        if not by_size[new_size]:
            continue
        new_mask = random.choice(by_size[new_size])
        if new_mask == old_mask:
            continue
        # Ensure distinctness
        if new_mask in family_set:
            continue
        
        # Calculate delta score using precomputed comparability
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
            family_set.remove(old_mask)
            family_set.add(new_mask)
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1
    
    # Final polish: try all possible single-element replacements
    if time.time() < deadline - 5:
        family_set = set(best_family)
        for idx in range(m):
            if time.time() >= deadline - 5:
                break
            old_mask = best_family[idx]
            best_delta = 0
            best_new_mask = None
            
            # Try replacing with masks from small and large sizes
            for size in [0, 1, 2, 3, 9, 10, 11, 12]:
                candidates = [mask for mask in by_size[size] if mask not in family_set and mask != old_mask]
                random.shuffle(candidates)
                for new_mask in candidates[:100]:
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        other = best_family[j]
                        if is_comparable(old_mask, other):
                            delta -= 1
                        if is_comparable(new_mask, other):
                            delta += 1
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask
            
            if best_new_mask is not None and best_delta > 0:
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                best_family[idx] = best_new_mask
                best_score += best_delta
    
    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
