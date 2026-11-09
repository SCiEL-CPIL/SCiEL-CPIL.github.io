"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    
    # Precompute: for each pair of sets, determine which third sets would form a sunflower
    # A sunflower triple: A∩B = A∩C = B∩C
    # For fixed A, B: C must contain A∩B and be disjoint from (A∪B) \ (A∩B)
    # So C = (A∩B) ∪ T where T ⊆ [1,16] \ (A∪B), |C|=4
    
    # Strategy: greedy with efficient conflict detection
    # For each candidate set, check against all existing sets
    
    chosen = []
    chosen_set = set()
    
    # For efficiency, track for each existing pair what's forbidden
    # But that's O(n^2) memory. Instead, for each new candidate, check against all chosen.
    
    # Better: for each candidate C, and each pair (A,B) in chosen, check if A∩B = A∩C = B∩C
    # This is O(|chosen|^2) per candidate which is too slow.
    
    # Alternative: for each candidate C and each existing A, compute A∩C.
    # Then for the pair to form sunflower with B, we need B∩C = A∩C and A∩B = A∩C.
    # So for fixed C and A, the forbidden B are those where A∩B = A∩C AND B∩C = A∩C.
    
    # Even better: precompute for each set its elements as a bitmask
    # Use bit operations for fast intersection
    
    # Let's use integer bitmasks for speed
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    
    # For greedy: try to add sets in some order
    # Order by... maybe just try all and use efficient checking
    
    # For each candidate, check against all pairs in chosen
    # Optimize: for candidate C with mask c, and existing A with mask a:
    #   inter_ac = a & c
    #   For B to form sunflower: need (a & b) == inter_ac AND (b & c) == inter_ac
    #   So b must satisfy: b & a = inter_ac AND b & c = inter_ac
    #   This means b contains inter_ac, and b is disjoint from (a & ~c) and (c & ~a)
    #   i.e., b & (a | c) = inter_ac
    
    # So for fixed C and A, the forbidden B are those where b & (a|c) == inter_ac
    # We can precompute for each existing B its mask, and check this condition.
    
    # But checking all pairs (A,B) for each C is still O(|chosen|^2).
    
    # Let me try a different approach: randomization + local search
    # Or: greedy with a good ordering
    
    # Actually, let's just do greedy with bitmasks and hope it's fast enough
    # |chosen| will be at most 250, so |chosen|^2 = 62500 per candidate
    # 1820 candidates * 62500 = 113M operations... might be too slow in Python
    
    # Let me try: for each candidate, only check against recently added or use a different strategy
    
    # Alternative: use the fact that if we build greedily, we can maintain a data structure
    # For each pair of "intersection patterns", track conflicts
    
    # Simpler: just do it with numpy for vectorized operations
    
    import numpy as np
    
    # Convert all masks to numpy array
    mask_array = np.array(masks, dtype=np.int64)
    
    # Greedy with vectorized checking
    # For candidate c, and all existing a_i, compute inter_ac_i = a_i & c
    # Then for each pair (i,j), check if a_i & a_j == inter_ac_i AND a_j & c == inter_ac_i
    # This is still O(n^2) but we can vectorize the first part
    
    # Let me try a smarter greedy: order candidates by some heuristic
    # and use early termination
    
    chosen_indices = []
    chosen_masks = []
    
    # Try all candidates in order, use vectorized check against chosen
    for idx, c in enumerate(all_sets):
        if time.time() > deadline:
            break
        cm = masks[idx]
        
        if not chosen_masks:
            chosen_indices.append(idx)
            chosen_masks.append(cm)
            continue
        
        # Vectorized: compute intersections of c with all chosen
        chosen_arr = np.array(chosen_masks, dtype=np.int64)
        inter_with_chosen = chosen_arr & cm  # shape (len(chosen),)
        
        # For each pair (i, j) in chosen, check sunflower condition
        # Need: chosen[i] & chosen[j] == inter_with_chosen[i] AND chosen[j] & cm == inter_with_chosen[i]
        # The second condition: chosen[j] & cm == inter_with_chosen[i]
        # But inter_with_chosen[i] = chosen[i] & cm
        # So we need chosen[j] & cm == chosen[i] & cm for all j where chosen[i] & chosen[j] == chosen[i] & cm
        
        # Let me check: for each i, for each j != i:
        #   if (chosen[i] & chosen[j]) == inter_with_chosen[i] and (chosen[j] & cm) == inter_with_chosen[i]:
        #       conflict
        
        # Vectorize: for each i, compute mask_i = chosen[i]
        # inter_ij = mask_i & chosen_arr  # all intersections with i
        # condition1 = (inter_ij == inter_with_chosen[i])
        # condition2 = (inter_with_chosen == inter_with_chosen[i])  # chosen[j] & cm == inter_with_chosen[i]
        # conflict if condition1 & condition2 & (j != i)
        
        # This is O(n^2) but with numpy it might be fast for n~250
        
        conflict = False
        n_chosen = len(chosen_masks)
        for i in range(n_chosen):
            if conflict:
                break
            inter_i_c = inter_with_chosen[i]
            # Check all j
            inter_i_j = chosen_arr[i] & chosen_arr  # shape (n_chosen,)
            cond1 = (inter_i_j == inter_i_c)
            cond2 = (inter_with_chosen == inter_i_c)
            # Exclude j == i
            cond2[i] = False
            if np.any(cond1 & cond2):
                conflict = True
                break
        
        if not conflict:
            chosen_indices.append(idx)
            chosen_masks.append(cm)
    
    return [sorted(all_sets[i]) for i in chosen_indices]
