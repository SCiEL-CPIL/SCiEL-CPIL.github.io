import time
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute neighbor masks
    # idx = sum(digit * q^i)
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

    neighbor_masks = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 1 << i
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
        neighbor_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    def greedy_construct_bitmask(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            uncovered_mask = all_mask ^ covered_mask
            # Find best candidate
            max_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                # Skip if already in code? No, greedy set cover doesn't strictly need to, 
                # but adding same word twice is useless. 
                # We can just check gain. If gain is 0, skip.
                gain = (neighbor_masks[i] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if not all covered
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= neighbor_masks[chosen]
            
        return code

    def reduce_code_bitmask(code):
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
            
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if c is essential
                # Remove c, see if all_mask is still covered
                temp_mask = covered_mask & ~neighbor_masks[c]
                if temp_mask != all_mask:
                    # c is essential, keep it
                    new_code.append(c)
                else:
                    # c is not essential, remove it
                    covered_mask = temp_mask
                    changed = True
            code = new_code
        return code

    def local_search_bitmask(code, deadline):
        code = list(code)
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                # Try removing c_remove
                temp_mask = covered_mask & ~neighbor_masks[c_remove]
                
                if temp_mask == all_mask:
                    # Can remove
                    covered_mask = temp_mask
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Find replacement
                    # Uncovered words are those in temp_mask that are not in all_mask? 
                    # No, uncovered = all_mask ^ temp_mask
                    uncovered = all_mask ^ temp_mask
                    # We need a word w such that neighbor_masks[w] covers all 'uncovered'
                    # i.e., (neighbor_masks[w] & uncovered) == uncovered
                    # And w != c_remove
                    
                    # To speed up, we can check candidates that have high overlap with uncovered
                    # Or just iterate all? 2187 iterations is fast.
                    best_w = -1
                    for w in range(total_words):
                        if w == c_remove:
                            continue
                        if (neighbor_masks[w] & uncovered) == uncovered:
                            # Check if w is already in code? 
                            # If w is in code, swapping is useless (or just moving).
                            # If w is not in code, it's a valid swap.
                            # Actually, if w is in code, it's already covered, so it doesn't help cover 'uncovered' 
                            # unless 'uncovered' is empty, which it isn't.
                            # So we just need w to cover 'uncovered'.
                            best_w = w
                            break # Take first one? Or random?
                            
                    if best_w != -1:
                        # Perform swap
                        covered_mask = temp_mask | neighbor_masks[best_w]
                        code[i] = best_w
                        improved = True
                        break
                        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_bitmask(seed)
        code = reduce_code_bitmask(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_bitmask(code, deadline - 5)
                code = reduce_code_bitmask(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
