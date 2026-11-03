import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Build cover matrix
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)

    # reverse_cover[w] = list of codeword indices whose cover set contains word w
    # This lets us find candidates covering a set of words by intersecting lists.
    reverse_cover = [None] * total_space
    for i in range(total_space):
        reverse_cover[i] = np.where(cover_matrix[i])[0]

    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            gains = cm_int8.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = 0
            
        return code

    def find_candidates_covering(unique_indices, exclude_set):
        # Intersect reverse_cover lists for the unique words, then exclude current codewords.
        if len(unique_indices) == 0:
            return np.arange(total_space, dtype=np.int64)
        # Start with the smallest list for efficiency
        lists = [reverse_cover[w] for w in unique_indices]
        lists.sort(key=len)
        cur = lists[0]
        for lst in lists[1:]:
            cur = np.intersect1d(cur, lst)
            if cur.size == 0:
                break
        if cur.size == 0:
            return cur
        if exclude_set:
            mask = np.ones(total_space, dtype=bool)
            for j in exclude_set:
                mask[j] = False
            cur = cur[mask[cur]]
        return cur

    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Update cover_count
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline - 2.0:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    # Remove
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try 1-for-1 swap
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        if len(unique_indices) > 0:
                            exclude_set = set(current_code)
                            candidates = find_candidates_covering(unique_indices, exclude_set)
                        
                            if len(candidates) > 0:
                                new_idx = candidates[random.randint(0, len(candidates)-1)]
                                
                                # Update cover_count
                                cover_count -= mask
                                cover_count += cover_matrix[new_idx]
                                
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                            else:
                                # Try 2-for-1: find j such that union of unique words of i and j
                                # can be covered by a single new word.
                                for j in range(n_cw):
                                    if time.time() > deadline - 2.0:
                                        return current_code
                                    if j == i:
                                        continue
                                    idx2 = current_code[j]
                                    mask2 = cover_matrix[idx2]
                                    unique_mask2 = mask2 & (cover_count == 1)
                                    if not np.any(unique_mask2):
                                        continue
                                    unique_indices2 = np.where(unique_mask2)[0]
                                    # combined unique set
                                    combined = np.unique(np.concatenate([unique_indices, unique_indices2]))
                                    exclude_set2 = set(current_code)
                                    candidates2 = find_candidates_covering(combined, exclude_set2)
                                    if len(candidates2) > 0:
                                        new_idx = candidates2[random.randint(0, len(candidates2)-1)]
                                        # remove i and j, add new_idx
                                        cover_count -= mask
                                        cover_count -= mask2
                                        cover_count += cover_matrix[new_idx]
                                        # remove both from list (j first if j>i to keep index valid)
                                        if j < i:
                                            current_code.pop(i)
                                            current_code.pop(j)
                                        else:
                                            current_code.pop(j)
                                            current_code.pop(i)
                                        current_code.append(int(new_idx))
                                        improved = True
                                        break
                                if improved:
                                    break
                            
        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    while time.time() < deadline - 12.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search on best_code with remaining time
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
