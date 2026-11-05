import time
import random
import math
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = 45.0  # Safety margin
    
    # Precompute conflict masks for fast validity checking
    # conflict_mask[v] is a bitmask where bit i is set if v conflicts with element i+1
    # Two elements conflict if they form a 3-AP with some third element
    # Actually, let's precompute for each pair (a,b) whether they can coexist
    # Better: for each v, precompute which elements in S would conflict with v
    
    # Precompute: for each v, the set of elements that conflict with v
    # v conflicts with a if there exists b such that {a, b, v} is a 3-AP
    # This happens when:
    # 1. v is middle: a + b = 2v, so b = 2v - a. Conflict if b in [1,n] and b != a, b != v
    # 2. v is end: a + v = 2b, so b = (a+v)/2. Conflict if (a+v) even, b in [1,n], b != a, b != v
    #    OR v + b = 2a, so b = 2a - v. Conflict if b in [1,n], b != a, b != v
    
    # For efficiency, precompute for each v a bitmask of conflicting elements
    # Use 0-indexed: element x corresponds to bit (x-1)
    conflict_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        mask = 0
        for a in range(1, n + 1):
            if a == v:
                continue
            # Case 1: v is middle, a is one end, c = 2v - a is other end
            c = 2 * v - a
            if 1 <= c <= n and c != a and c != v:
                # a and c conflict via v, but we want: does a conflict with v?
                # Yes, if c exists in [1,n] and c != a, c != v
                # But actually, a conflicts with v if there's some c such that {a,c,v} is AP
                # Here c = 2v - a, so if c is valid, then a conflicts with v
                mask |= (1 << (a - 1))
                break  # No need to check further for this a
            
            # Case 2: v is end
            # Subcase 2a: a, b, v where b = (a+v)/2
            if (a + v) % 2 == 0:
                b = (a + v) // 2
                if 1 <= b <= n and b != a and b != v:
                    mask |= (1 << (a - 1))
                    break
            
            # Subcase 2b: v, b, a where b = (v+a)/2 - same as above
            # Already covered
        
        # Wait, I need to be more careful. Let me reconsider.
        # Two elements a and v conflict if there exists b such that {a, b, v} forms a 3-AP
        # The three elements in a 3-AP are x, x+d, x+2d for some d != 0
        # So {a, b, v} is a 3-AP if they can be ordered as x, x+d, x+2d
        # This means one of them is the middle: 2*middle = end1 + end2
        
        # Let me redo this properly
        conflict_mask = [0] * (n + 1)
        for v in range(1, n + 1):
            mask = 0
            for a in range(1, n + 1):
                if a == v:
                    continue
                # Check if a and v can be part of a 3-AP
                # They can if there exists b such that {a, b, v} is a 3-AP
                # b = 2v - a (v is middle)
                b1 = 2 * v - a
                if 1 <= b1 <= n and b1 != a and b1 != v:
                    mask |= (1 << (a - 1))
                    continue
                # b = (a + v) / 2 (v is an end)
                if (a + v) % 2 == 0:
                    b2 = (a + v) // 2
                    if 1 <= b2 <= n and b2 != a and b2 != v:
                        mask |= (1 << (a - 1))
                        continue
                # b = 2a - v (a is middle)
                b3 = 2 * a - v
                if 1 <= b3 <= n and b3 != a and b3 != v:
                    mask |= (1 << (a - 1))
                    continue
            conflict_mask[v] = mask
    
    def is_valid_addition_bitmask(current_mask, v):
        """Check if v can be added to set represented by current_mask"""
        # v conflicts if any element in current_mask conflicts with v
        # conflict_mask[v] has bits set for elements that conflict with v
        return (current_mask & conflict_mask[v]) == 0
    
    def get_score_from_mask(mask):
        """Compute sum of reciprocals from bitmask"""
        score = 0.0
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                score += 1.0 / x
        return score
    
    def mask_to_set(mask):
        """Convert bitmask to set"""
        s = set()
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                s.add(x)
        return s
    
    def set_to_mask(s):
        """Convert set to bitmask"""
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        return mask
    
    # 1. Greedy Initialization using bitmask
    current_mask = 0
    for x in range(1, n + 1):
        if is_valid_addition_bitmask(current_mask, x):
            current_mask |= (1 << (x - 1))
    
    best_mask = current_mask
    best_score = get_score_from_mask(best_mask)
    
    # 2. Local Search with bitmask
    no_improvement_count = 0
    
    while time.time() - start_time < deadline:
        improved = False
        
        # Try 1-swaps: remove u, add v where v < u (to improve reciprocal sum)
        in_S = [x for x in range(1, n + 1) if current_mask & (1 << (x - 1))]
        not_in_S = [x for x in range(1, n + 1) if not (current_mask & (1 << (x - 1)))]
        
        for u in in_S:
            for v in not_in_S:
                if v >= u:
                    break  # Since not_in_S is sorted, no more v < u
                
                # Check if removing u and adding v is valid
                test_mask = current_mask & ~(1 << (u - 1))
                if is_valid_addition_bitmask(test_mask, v):
                    new_mask = test_mask | (1 << (v - 1))
                    new_score = best_score - 1.0 / u + 1.0 / v
                    if new_score > best_score + 1e-9:
                        current_mask = new_mask
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try 2-swaps
            in_S = [x for x in range(1, n + 1) if current_mask & (1 << (x - 1))]
            not_in_S = [x for x in range(1, n + 1) if not (current_mask & (1 << (x - 1)))]
            
            found_2swap = False
            for i, u1 in enumerate(in_S):
                if found_2swap:
                    break
                for u2 in in_S[i+1:]:
                    if found_2swap:
                        break
                    if len(in_S) <= 3:
                        continue
                    
                    test_mask = current_mask & ~(1 << (u1 - 1)) & ~(1 << (u2 - 1))
                    
                    for v1 in not_in_S:
                        if found_2swap:
                            break
                        if is_valid_addition_bitmask(test_mask, v1):
                            test_mask2 = test_mask | (1 << (v1 - 1))
                            for v2 in not_in_S:
                                if v2 == v1:
                                    continue
                                if not is_valid_addition_bitmask(test_mask2, v2):
                                    continue
                                
                                new_score = best_score - 1.0 / u1 - 1.0 / u2 + 1.0 / v1 + 1.0 / v2
                                if new_score > best_score + 1e-9:
                                    current_mask = test_mask2 | (1 << (v2 - 1))
                                    best_score = new_score
                                    improved = True
                                    found_2swap = True
                                    break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 3:
                # Perturbation: remove 2 random elements and re-greedy
                perturbed_mask = current_mask
                if bin(perturbed_mask).count('1') > 3:
                    elems = [x for x in range(1, n + 1) if perturbed_mask & (1 << (x - 1))]
                    random.shuffle(elems)
                    removed = elems[:2]
                    for r in removed:
                        perturbed_mask &= ~(1 << (r - 1))
                    
                    # Re-greedy
                    for x in range(1, n + 1):
                        if not (perturbed_mask & (1 << (x - 1))):
                            if is_valid_addition_bitmask(perturbed_mask, x):
                                perturbed_mask |= (1 << (x - 1))
                    
                    new_score = get_score_from_mask(perturbed_mask)
                    if new_score > best_score:
                        current_mask = perturbed_mask
                        best_score = new_score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 5
    
    return list(mask_to_set(best_mask))
