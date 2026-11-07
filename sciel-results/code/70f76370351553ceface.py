"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 12
M = 40
FULL_MASK = (1 << N) - 1
ALL_MASKS = list(range(1 << N))

def _count_comparable_pairs_masks(masks):
    """Count unordered pairs where one mask is strict subset of the other."""
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if mi == mj:
                continue
            if (mi & mj) == mi or (mi & mj) == mj:
                count += 1
    return count

def _masks_to_sets(masks):
    """Convert list of bitmasks to list of lists of ints."""
    result = []
    for mask in masks:
        s = []
        for b in range(N):
            if mask & (1 << b):
                s.append(b)
        result.append(s)
    return result

def _construct():
    deadline = time.time() + 85.0
    random.seed(42)
    
    best_score = -1
    best_family = None
    
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy construction with bitmask representation
        family = [0, FULL_MASK]  # empty set and full set
        family_set = set(family)
        
        while len(family) < M:
            best_mask = None
            best_gain = -1
            # Sample candidates
            candidates = [m for m in ALL_MASKS if m not in family_set]
            random.shuffle(candidates)
            sample = candidates[:min(300, len(candidates))]
            
            for s in sample:
                gain = 0
                for t in family:
                    if s == t:
                        continue
                    if (s & t) == s or (s & t) == t:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            
            if best_mask is not None:
                family.append(best_mask)
                family_set.add(best_mask)
            else:
                s = random.choice(candidates)
                family.append(s)
                family_set.add(s)
        
        score = _count_comparable_pairs_masks(family)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        # Local search refinement
        for iteration in range(1000):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try replacing one set
            for i in range(M):
                if time.time() > deadline - 5.0:
                    break
                current = family[i]
                # Compute current contribution of family[i]
                current_contrib = 0
                for j in range(M):
                    if j == i:
                        continue
                    t = family[j]
                    if current == t:
                        continue
                    if (current & t) == current or (current & t) == t:
                        current_contrib += 1
                
                # Try a few random replacements
                for _ in range(8):
                    candidate = random.choice(ALL_MASKS)
                    if candidate in family_set or candidate == current:
                        continue
                    
                    # Compute new contribution of candidate
                    new_contrib = 0
                    for j in range(M):
                        if j == i:
                            continue
                        t = family[j]
                        if candidate == t:
                            continue
                        if (candidate & t) == candidate or (candidate & t) == t:
                            new_contrib += 1
                    
                    delta = new_contrib - current_contrib
                    if delta > 0:
                        family_set.discard(current)
                        family[i] = candidate
                        family_set.add(candidate)
                        score += delta
                        improved = True
                        break
            
            if not improved:
                # Try pairwise swaps
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    i, j = random.sample(range(M), 2)
                    cand_i = random.choice(ALL_MASKS)
                    cand_j = random.choice(ALL_MASKS)
                    if cand_i in family_set or cand_j in family_set:
                        continue
                    if cand_i == cand_j or cand_i == family[i] or cand_j == family[j]:
                        continue
                    
                    # Compute delta for this swap
                    old_ij_comp = 0
                    if family[i] != family[j]:
                        a, b = family[i], family[j]
                        if (a & b) == a or (a & b) == b:
                            old_ij_comp = 1
                    new_ij_comp = 0
                    if cand_i != cand_j:
                        a, b = cand_i, cand_j
                        if (a & b) == a or (a & b) == b:
                            new_ij_comp = 1
                    
                    delta = 0
                    for k in range(M):
                        if k == i or k == j:
                            continue
                        fk = family[k]
                        old_k = 0
                        if family[i] != fk:
                            a, b = family[i], fk
                            if (a & b) == a or (a & b) == b:
                                old_k += 1
                        if family[j] != fk:
                            a, b = family[j], fk
                            if (a & b) == a or (a & b) == b:
                                old_k += 1
                        new_k = 0
                        if cand_i != fk:
                            a, b = cand_i, fk
                            if (a & b) == a or (a & b) == b:
                                new_k += 1
                        if cand_j != fk:
                            a, b = cand_j, fk
                            if (a & b) == a or (a & b) == b:
                                new_k += 1
                        delta += new_k - old_k
                    delta += new_ij_comp - old_ij_comp
                    
                    if delta > 0:
                        family_set.discard(family[i])
                        family_set.discard(family[j])
                        family[i] = cand_i
                        family[j] = cand_j
                        family_set.add(cand_i)
                        family_set.add(cand_j)
                        score += delta
                        improved = True
                        break
            
            if not improved:
                break
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    # Final polish on best_family
    if best_family is not None:
        family = best_family[:]
        family_set = set(family)
        score = _count_comparable_pairs_masks(family)
        
        for iteration in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            for i in range(M):
                if time.time() > deadline - 2.0:
                    break
                current = family[i]
                current_contrib = 0
                for j in range(M):
                    if j == i:
                        continue
                    t = family[j]
                    if current == t:
                        continue
                    if (current & t) == current or (current & t) == t:
                        current_contrib += 1
                
                for _ in range(12):
                    candidate = random.choice(ALL_MASKS)
                    if candidate in family_set or candidate == current:
                        continue
                    
                    new_contrib = 0
                    for j in range(M):
                        if j == i:
                            continue
                        t = family[j]
                        if candidate == t:
                            continue
                        if (candidate & t) == candidate or (candidate & t) == t:
                            new_contrib += 1
                    
                    delta = new_contrib - current_contrib
                    if delta > 0:
                        family_set.discard(current)
                        family[i] = candidate
                        family_set.add(candidate)
                        score += delta
                        improved = True
                        break
            
            if not improved:
                break
        
        if score > best_score:
            best_score = score
            best_family = family[:]
    
    return _masks_to_sets(best_family)

def construct():
    result = _construct()
    # Ensure all elements are Python ints and sequences are lists
    cleaned = []
    for s in result:
        cleaned.append([int(x) for x in s])
    return cleaned
# EVOLVE-BLOCK-END
