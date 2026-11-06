"""Deterministic greedy with multiple structured orderings."""
# EVOLVE-BLOCK-START
import itertools
import random

def _generate_conflicts(K, R, mask_to_idx):
    """
    Generate indices of all 4-masks X such that:
    1. X contains K
    2. X is disjoint from R
    """
    universe_mask = (1 << 16) - 1
    allowed_mask = universe_mask & ~R
    
    # We need to choose (4 - popcount(K)) elements from (allowed_mask & ~K)
    avail_elements = []
    m = allowed_mask & ~K
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        avail_elements.append(idx)
        m ^= lsb
        
    k_size = bin(K).count("1")
    num_to_choose = 4 - k_size
    
    conflicts = []
    if num_to_choose < 0 or num_to_choose > len(avail_elements):
        return conflicts
        
    for comb in itertools.combinations(avail_elements, num_to_choose):
        x_mask = K
        for i in comb:
            x_mask |= (1 << i)
            
        if x_mask in mask_to_idx:
            conflicts.append(mask_to_idx[x_mask])
            
    return conflicts

def _greedy_fast(ordering, all_masks, mask_to_idx, rng=None):
    if rng is None:
        rng = random.Random(42)
        
    candidates = list(all_masks)
    if ordering == "random":
        rng.shuffle(candidates)
    elif ordering == "lex":
        candidates.sort()
    elif ordering == "sum":
        def get_sum(m):
            s = 0
            mm = m
            while mm:
                lsb = mm & -mm
                s += lsb.bit_length()
                mm ^= lsb
            return s
        candidates.sort(key=get_sum)
        
    conflict_counts = [0] * len(all_masks)
    family = []
    
    for s_mask in candidates:
        s_idx = mask_to_idx[s_mask]
        if conflict_counts[s_idx] > 0:
            continue
            
        family.append(s_mask)
        
        for a_mask in family[:-1]:
            K = a_mask & s_mask
            R = (a_mask ^ K) | (s_mask ^ K)
            
            conflicts = _generate_conflicts(K, R, mask_to_idx)
            for c_idx in conflicts:
                conflict_counts[c_idx] += 1
                
    return family

def construct():
    all_sets = list(itertools.combinations(range(1, 17), 4))
    all_masks = []
    mask_to_idx = {}
    
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        all_masks.append(m)
        mask_to_idx[m] = i
        
    best_family = []
    best_size = 0
    
    for i in range(100):
        family = _greedy_fast("random", all_masks, mask_to_idx, random.Random(i))
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
    for ordering in ["lex", "sum"]:
        family = _greedy_fast(ordering, all_masks, mask_to_idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
    result = []
    for m in best_family:
        s = []
        mm = m
        while mm:
            lsb = mm & -mm
            s.append(lsb.bit_length())
            mm ^= lsb
        result.append(sorted(s))
        
    return result
# EVOLVE-BLOCK-END
