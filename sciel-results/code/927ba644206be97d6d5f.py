import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    all_set = set(all_masks)
    
    def get_forbidden_for_pair(A, B):
        """Return all C such that A∩B = A∩C = B∩C, with |C|=4."""
        S = A & B
        s_size = bin(S).count('1')
        remaining = 4 - s_size
        if remaining < 0:
            return []
        # C must contain S, and C\S is a subset of [16]\(A|B) of size 'remaining'
        complement = ~(A | B) & 0xFFFF  # 16 bits
        # Find all subsets of complement of size 'remaining'
        bits = [i for i in range(16) if complement & (1 << i)]
        if remaining == 0:
            return [S]
        if remaining > len(bits):
            return []
        result = []
        for combo in combinations(bits, remaining):
            T = 0
            for b in combo:
                T |= (1 << b)
            C = S | T
            result.append(C)
        return result
    
    def check_valid_addition(C, family_list):
        """Check if C can be added to family (list of masks)."""
        n = len(family_list)
        for i in range(n):
            A = family_list[i]
            for j in range(i + 1, n):
                B = family_list[j]
                I = A & B
                if (A & C) == I and (B & C) == I:
                    return False
        return True
    
    # Strategy: Use a more efficient greedy approach with better ordering
    # and local search improvements.
    
    best_family = []
    best_size = 0
    
    # Multiple restarts with different orderings
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            # Random shuffle
            shuffled = all_masks[:]
            random.shuffle(shuffled)
        elif trial % 3 == 1:
            # Sort by "diversity" - prefer sets with low overlap with existing
            shuffled = all_masks[:]
            random.shuffle(shuffled)
            # Weight by something... let's just use random for now
        else:
            # Another random shuffle
            shuffled = all_masks[:]
            random.shuffle(shuffled)
        
        family = []
        family_set = set()
        forbidden = set()
        
        for mask in shuffled:
            if time.time() > deadline - 10:
                break
            if mask in forbidden:
                continue
            if mask in family_set:
                continue
            
            # Quick check: can we add this?
            # The forbidden set already captures most conflicts, but let's be safe
            # Actually, if mask is not in forbidden, it should be valid
            # But we need to verify because forbidden is built incrementally
            
            # Add to family
            family.append(mask)
            family_set.add(mask)
            
            # Update forbidden set: for each existing family member f,
            # add all C that form sunflower with (mask, f)
            for f in family[:-1]:
                for C in get_forbidden_for_pair(mask, f):
                    forbidden.add(C)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search: try to improve the best family found
    family = best_family[:]
    family_set = set(family)
    
    # Rebuild forbidden set for current family
    forbidden = set()
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            for C in get_forbidden_for_pair(family[i], family[j]):
                forbidden.add(C)
    
    # Try to add any mask not in family and not forbidden
    added = True
    while added and time.time() < deadline - 8:
        added = False
        for mask in all_masks:
            if mask not in family_set and mask not in forbidden:
                family.append(mask)
                family_set.add(mask)
                # Update forbidden
                for f in family[:-1]:
                    for C in get_forbidden_for_pair(mask, f):
                        forbidden.add(C)
                added = True
    
    # Simulated annealing / swap-based local search
    if time.time() < deadline - 5:
        T = 1.0
        cooling = 0.999
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2 and T > 0.05 and no_improve_count < max_no_improve:
            # Pick a random mask to remove
            idx_out = random.randint(0, len(family) - 1)
            mask_out = family[idx_out]
            
            # Pick a random mask to add (not in family)
            candidates = [m for m in all_masks if m not in family_set]
            if not candidates:
                break
            mask_in = random.choice(candidates)
            
            # Try the swap: remove mask_out, add mask_in
            # Check if mask_in is valid with family minus mask_out
            fam_temp = family[:idx_out] + family[idx_out + 1:]
            
            valid = True
            for i in range(len(fam_temp)):
                A = fam_temp[i]
                for j in range(i + 1, len(fam_temp)):
                    B = fam_temp[j]
                    I = A & B
                    if (A & mask_in) == I and (B & mask_in) == I:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Accept the swap
                family = fam_temp + [mask_in]
                family_set.remove(mask_out)
                family_set.add(mask_in)
                no_improve_count = 0
                
                # Rebuild forbidden (expensive but necessary for correctness)
                forbidden = set()
                for i in range(len(family)):
                    for j in range(i + 1, len(family)):
                        for C in get_forbidden_for_pair(family[i], family[j]):
                            forbidden.add(C)
            else:
                no_improve_count += 1
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for mask in all_masks:
                if mask not in family_set and mask not in forbidden:
                    family.append(mask)
                    family_set.add(mask)
                    for f in family[:-1]:
                        for C in get_forbidden_for_pair(mask, f):
                            forbidden.add(C)
                    added = True
    
    # Convert back to sorted lists
    result = []
    for mask in family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result
