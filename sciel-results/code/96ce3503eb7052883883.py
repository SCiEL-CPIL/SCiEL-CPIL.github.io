import time
import random
import itertools

def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0
    
    total = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for x in w:
            idx = idx * q + x
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute balls as bitmasks
    # ball_mask[i] is an int with bits set for all words within distance 1 of i
    ball_masks = [0] * total
    for idx in range(total):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v != orig:
                    w[i] = v
                    n_idx = word_to_idx(w)
                    mask |= (1 << n_idx)
            w[i] = orig
        ball_masks[idx] = mask
        
    full_mask = (1 << total) - 1
    
    # Greedy Set Cover to get initial code
    code = []
    covered_mask = 0
    uncovered_count = total
    
    # Precompute gain for each word? 
    # gain[i] = number of bits in ball_masks[i] not in covered_mask
    # We can just compute it on the fly or maintain it.
    # For simplicity and speed, let's just iterate.
    
    # To speed up greedy, we can precompute the list of words to consider.
    # But N=729 is small.
    
    while uncovered_count > 0:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            if i in code: # Optimization: skip if already in code? 
                # Actually, we should pick the best from ALL, but adding one already in code is useless.
                # So we check gain for all, but ignore if in code? 
                # No, if i is in code, its gain is 0 effectively for adding.
                pass
            gain = bin(ball_masks[i] & ~covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break # Should not happen if uncovered > 0
            
        code.append(best_idx)
        covered_mask |= ball_masks[best_idx]
        # Update uncovered_count
        # uncovered_count = total - bin(covered_mask).count('1')
        # This is O(N) per step. 729 * 729 steps worst case? No, code size is ~75.
        # So 75 * 729 * count('1') is fine.
        
    # Initial code size is likely ~80-90.
    
    # Local Search
    def is_valid(mask):
        return mask == full_mask
        
    def get_union_mask(code_list):
        m = 0
        for c in code_list:
            m |= ball_masks[c]
        return m
        
    def try_delete_redundant(code_list, union_m):
        """
        Returns (new_code, new_union, deleted_count)
        """
        deleted = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Check if c is redundant
                # others = union_m ^ ball_masks[c] is WRONG if bits overlap.
                # We need OR of all others.
                others_m = 0
                for j in range(len(code_list)):
                    if j != i:
                        others_m |= ball_masks[code_list[j]]
                if (others_m & ball_masks[c]) == ball_masks[c]:
                    # Redundant
                    code_list.pop(i)
                    # Update union_m
                    # union_m is OR of all. Removing c might not change union_m if others cover it.
                    # Since it is redundant, union_m remains the same.
                    deleted += 1
                    changed = True
                    break # Restart loop
        return code_list, union_m, deleted

    # Main Local Search Loop
    current_code = list(code)
    current_union = get_union_mask(current_code)
    
    # Initial cleanup
    current_code, current_union, _ = try_delete_redundant(current_code, current_union)
    
    rng = random.Random(42)
    best_size = len(current_code)
    best_code = list(current_code)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # 1. Try to delete
        new_code, new_union, deleted = try_delete_redundant(list(current_code), current_union)
        if deleted > 0:
            current_code = new_code
            current_union = new_union
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue # Restart loop
        
        # 2. Try swaps
        swap_found = False
        # To speed up, we can randomize the order of c_old or c_new
        c_old_list = list(current_code)
        rng.shuffle(c_old_list)
        
        for c_old in c_old_list:
            # If time is tight, break
            if time.time() > deadline - 5:
                break
                
            # Candidates for c_new: all words not in current_code
            # We can iterate all and skip those in code
            c_new_list = [i for i in range(total) if i not in current_code]
            rng.shuffle(c_new_list) # Randomize to escape local minima
            
            for c_new in c_new_list:
                # Check validity of swap
                # New set: current_code - {c_old} + {c_new}
                # We can compute new_union efficiently?
                # new_union = 0
                # for c in current_code:
                #    if c != c_old: new_union |= ball_masks[c]
                # new_union |= ball_masks[c_new]
                
                # Optimization: 
                # We know current_union.
                # If we remove c_old, we lose bits that ONLY c_old covers.
                # But we add c_new.
                # Just recompute is safer and fast enough.
                
                new_union = 0
                for c in current_code:
                    if c != c_old:
                        new_union |= ball_masks[c]
                new_union |= ball_masks[c_new]
                
                if new_union == full_mask:
                    # Valid swap
                    # Try to delete redundant nodes from this new set
                    temp_code = [c for c in current_code if c != c_old]
                    temp_code.append(c_new)
                    temp_union = new_union
                    
                    # Try delete
                    temp_code, temp_union, del_count = try_delete_redundant(temp_code, temp_union)
                    
                    if len(temp_code) < len(current_code):
                        # Improvement!
                        current_code = temp_code
                        current_union = temp_union
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        swap_found = True
                        break # Break c_new loop
                    elif len(temp_code) == len(current_code):
                        # Same size, but maybe different structure. 
                        # Accept with small probability to explore?
                        # Or just accept if it allows future improvements?
                        # Let's accept 10% of the time to diversify.
                        if rng.random() < 0.1:
                            current_code = temp_code
                            current_union = temp_union
                            swap_found = True
                            break
                            
            if swap_found:
                break # Break c_old loop
                
        if not swap_found:
            # Stuck. Perturb.
            # Remove 2 random codewords and repair?
            # Or just random restart?
            # Let's try a small perturbation: remove 1 random, repair with greedy.
            if len(current_code) > 10:
                idx_to_remove = rng.randint(0, len(current_code) - 1)
                current_code.pop(idx_to_remove)
                current_union = get_union_mask(current_code)
                
                # Greedy repair
                while current_union != full_mask:
                    best_idx = -1
                    best_gain = 0
                    for i in range(total):
                        if i in current_code:
                            continue
                        gain = bin(ball_masks[i] & ~current_union).count('1')
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    current_code.append(best_idx)
                    current_union |= ball_masks[best_idx]
                    
                # Cleanup
                current_code, current_union, _ = try_delete_redundant(current_code, current_union)

    return [idx_to_word(i) for i in best_code]
