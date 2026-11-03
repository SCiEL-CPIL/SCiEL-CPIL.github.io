import time
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute index conversions
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighborhood masks
    # Mask[i] is a bitset of words covered by codeword i
    # PreMask[j] is a bitset of codewords that cover word j
    masks = [0] * total_words
    pre_masks = [0] * total_words
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 1 << i
        # Add i itself
        # Add neighbors
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
        masks[i] = mask
        
    for j in range(total_words):
        pm = 0
        # PreMask[j] should contain i if j is in masks[i]
        # It is faster to iterate i and set bits in pre_masks?
        # Or just check membership.
        # Since total_words is 2187, iterating all i for each j is 2187*2187 ~ 4.7M ops. Fast enough.
        for i in range(total_words):
            if (masks[i] >> j) & 1:
                pm |= (1 << i)
        pre_masks[j] = pm

    all_mask = (1 << total_words) - 1
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            uncovered_mask = all_mask ^ covered_mask
            max_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (masks[i] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= masks[chosen]
            
        return code

    def reduce_code(code):
        code = list(code)
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if c is essential
                # Compute coverage of others
                others_mask = 0
                for o in code:
                    if o != c:
                        others_mask |= masks[o]
                
                if (masks[c] & ~others_mask) == 0:
                    # c is redundant
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    def optimize_2_for_1(code):
        code = list(code)
        # Maintain counts
        counts = [0] * total_words
        for c in code:
            m = masks[c]
            # Add 1 to counts for all bits set in m
            # This is slow if done naively. 
            # But we only do this in the pair loop.
            # We will update counts incrementally in the pair check.
            
        improved = True
        while improved:
            improved = False
            # Reset counts for current code
            for c in range(total_words):
                counts[c] = 0
            for c in code:
                m = masks[c]
                # Iterate bits in m
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    mm ^= lsb
            
            n_code = len(code)
            for i_idx in range(n_code):
                if improved:
                    break
                for j_idx in range(i_idx + 1, n_code):
                    if improved:
                        break
                    
                    u = code[i_idx]
                    v = code[j_idx]
                    
                    # Decrement counts for u and v
                    # Save old counts to restore? Or just recompute?
                    # Recomputing is expensive. Let's store the changes.
                    changes_u = []
                    changes_v = []
                    
                    mu = masks[u]
                    mm = mu
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        changes_u.append(idx)
                        mm ^= lsb
                        
                    mv = masks[v]
                    mm = mv
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        changes_v.append(idx)
                        mm ^= lsb
                        
                    # Find uncovered words U
                    U = []
                    # Only words in changes_u or changes_v can have count 0
                    # Union of indices
                    check_set = set(changes_u + changes_v)
                    for idx in check_set:
                        if counts[idx] == 0:
                            U.append(idx)
                            
                    if not U:
                        # We can remove both u and v
                        # Remove from code
                        code.pop(j_idx) # Remove v
                        code.pop(i_idx) # Remove u
                        improved = True
                        # Restore counts? No, we broke the loop and will recompute at top of while
                        break
                    
                    # Check if there is a w covering U
                    # I = intersection of PreMask[k] for k in U
                    I = all_mask
                    for k in U:
                        I &= pre_masks[k]
                        
                    if I != 0:
                        # Find a candidate w in I
                        # Prefer w not in {u, v}
                        # Check if I has any bit other than u, v
                        # Mask out u and v
                        I_without_uv = I & ~(1 << u) & ~(1 << v)
                        
                        if I_without_uv != 0:
                            # Pick one
                            lsb = I_without_uv & -I_without_uv
                            w = lsb.bit_length() - 1
                            
                            # Replace u, v with w
                            code.pop(j_idx)
                            code.pop(i_idx)
                            code.append(w)
                            improved = True
                            break
                        else:
                            # Only u or v cover U. 
                            # If we pick u, code becomes C \ {v} (size - 1? No, C \ {u, v} U {u} = C \ {v}).
                            # This is a 1-for-1 swap in terms of set, but effectively removing v.
                            # Is v redundant given u?
                            # If U is covered by u, and other words cover the rest, then v is redundant?
                            # Yes, if we replace {u, v} with {u}, we remove v.
                            # Let's check if this is valid.
                            # The new code is C \ {v}.
                            # We already checked that C \ {u, v} covers everything except U.
                            # And u covers U.
                            # So C \ {v} covers everything.
                            # So we can just remove v.
                            code.pop(j_idx)
                            improved = True
                            break
                            
                    # Restore counts
                    for idx in changes_u:
                        counts[idx] += 1
                    for idx in changes_v:
                        counts[idx] += 1
                        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
             code = optimize_2_for_1(code)
             code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
