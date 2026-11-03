import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Generate all words as tuples of integers
    words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * total_words
    
    for i, w in enumerate(words):
        ball_val = 0
        # Center
        ball_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                ball_val |= (1 << j)
        balls[i] = ball_val
        
    best_code = None
    best_size = total_words + 1
    
    start_time = time.time()
    max_time = 110.0
    
    def greedy_construct():
        uncovered = (1 << total_words) - 1
        code = []
        code_indices = set()
        
        while uncovered:
            max_covered = 0
            best_candidates = []
            
            for i in range(total_words):
                if i in code_indices:
                    continue
                covered_count = (balls[i] & uncovered).bit_count()
                
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_candidates = [i]
                elif covered_count == max_covered and covered_count > 0:
                    best_candidates.append(i)
                    
            if max_covered == 0:
                break
                
            chosen_idx = random.choice(best_candidates)
            code.append(words[chosen_idx])
            code_indices.add(chosen_idx)
            uncovered &= ~balls[chosen_idx]
            
        return code, code_indices

    def local_search(code, code_indices):
        improved = True
        local_iter = 0
        max_local_iter = 100
        
        while improved and local_iter < max_local_iter and time.time() - start_time < max_time:
            improved = False
            local_iter += 1
            
            # Calculate counts
            counts = [0] * total_words
            for c in code:
                idx = word_to_idx[c]
                b = balls[idx]
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] += 1
                    temp ^= lsb
                    
            # 1. Try to remove redundant words
            for c in code:
                idx = word_to_idx[c]
                b = balls[idx]
                is_redundant = True
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        is_redundant = False
                        break
                    temp ^= lsb
                    
                if is_redundant:
                    # Remove
                    temp = b
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                    code.remove(c)
                    code_indices.remove(idx)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. Try 1-for-1 swaps
            swap_done = False
            for c in code:
                idx = word_to_idx[c]
                b = balls[idx]
                unique_mask = 0
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        unique_mask |= lsb
                    temp ^= lsb
                    
                if unique_mask == 0:
                    continue
                    
                # Find candidate to replace c
                lsb = unique_mask & -unique_mask
                w_idx = lsb.bit_length() - 1
                
                candidates = []
                for j in range(total_words):
                    if j in code_indices:
                        continue
                    if balls[j] & (1 << w_idx):
                        if (balls[j] & unique_mask) == unique_mask:
                            candidates.append(j)
                            
                if candidates:
                    new_idx = random.choice(candidates)
                    
                    # Swap
                    temp = b
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                    code.remove(c)
                    code_indices.remove(idx)
                    
                    b_new = balls[new_idx]
                    temp = b_new
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] += 1
                        temp ^= lsb
                    code.append(words[new_idx])
                    code_indices.add(new_idx)
                    
                    improved = True
                    swap_done = True
                    break
                    
            if swap_done:
                continue
                
            # 3. Try 2-for-1 swaps (Remove 2, Add 1)
            # This is more expensive. We sample pairs.
            if len(code) >= 2:
                # Sample a few pairs
                for _ in range(10):
                    if len(code) < 2: break
                    i1, i2 = random.sample(range(len(code)), 2)
                    c1 = code[i1]
                    c2 = code[i2]
                    idx1 = word_to_idx[c1]
                    idx2 = word_to_idx[c2]
                    
                    # Union of unique words of c1 and c2
                    unique_mask = 0
                    for idx in [idx1, idx2]:
                        b = balls[idx]
                        temp = b
                        while temp:
                            lsb = temp & -temp
                            idx_bit = lsb.bit_length() - 1
                            if counts[idx_bit] <= 2: # If count is 1 or 2, it's unique to this pair
                                # If count is 1, it's unique to one. If 2, unique to both.
                                # We need to cover all words that are ONLY covered by c1 or c2.
                                unique_mask |= lsb
                            temp ^= lsb
                            
                    if unique_mask == 0:
                        continue
                        
                    # Find a single ball c' that covers unique_mask
                    lsb = unique_mask & -unique_mask
                    w_idx = lsb.bit_length() - 1
                    
                    found = False
                    for j in range(total_words):
                        if j in code_indices:
                            continue
                        if balls[j] & (1 << w_idx):
                            if (balls[j] & unique_mask) == unique_mask:
                                # Perform 2-for-1 swap
                                # Remove c1, c2
                                for idx in [idx1, idx2]:
                                    b = balls[idx]
                                    temp = b
                                    while temp:
                                        lsb = temp & -temp
                                        idx_bit = lsb.bit_length() - 1
                                        counts[idx_bit] -= 1
                                        temp ^= lsb
                                code.remove(c1)
                                code.remove(c2)
                                code_indices.remove(idx1)
                                code_indices.remove(idx2)
                                
                                # Add c'
                                b_new = balls[j]
                                temp = b_new
                                while temp:
                                    lsb = temp & -temp
                                    idx_bit = lsb.bit_length() - 1
                                    counts[idx_bit] += 1
                                    temp ^= lsb
                                code.append(words[j])
                                code_indices.add(j)
                                
                                improved = True
                                found = True
                                break
                                
                    if found:
                        break
                        
        return code, code_indices

    while time.time() - start_time < max_time:
        code, code_indices = greedy_construct()
        code, code_indices = local_search(code, code_indices)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            if best_size <= 186:
                break

    return [list(c) for c in best_code]
