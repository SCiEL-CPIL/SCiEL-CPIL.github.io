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

    # Precompute cover matrix using NumPy for speed
    # cover_matrix[i, j] is True if word j is within Hamming distance <= radius of word i
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)

    if radius >= 1:
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = q ** (n - 1 - pos)
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute cover_matrix as int8 for fast dot products
    cover_matrix_int8 = cover_matrix.astype(np.int8)
    cover_matrix_uint8 = cover_matrix.astype(np.uint8)

    # Greedy Set Cover using NumPy
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int = uncovered.astype(np.int8)
        
        while np.any(uncovered):
            if time.time() > deadline - 2.0:
                break
            gains = cover_matrix_int8.dot(unc_int)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            unc_int[cover_matrix[best_idx]] = 0
        return code

    # Local Search: Remove redundant codewords and perform swaps
    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        if max_time is None:
            max_time = deadline

        current_code = code[:]
        code_set = set(current_code)

        # Build cover_count array
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix_int8[idx]

        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            code_set = set(current_code)
            n_cw = len(current_code)

            for i in range(n_cw):
                if time.time() > max_time:
                    return current_code

                idx = current_code[i]
                mask = cover_matrix[idx]

                # Check if removable: all covered words have count > 1
                counts_in_mask = cover_count[mask]
                if np.min(counts_in_mask) > 1:
                    # Decrement cover_count and remove
                    cover_count -= cover_matrix_int8[idx]
                    current_code.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break
                else:
                    # Try to find a swap
                    # Find words uniquely covered by idx
                    unique_mask = mask & (cover_count == 1)
                    if np.any(unique_mask):
                        # Find a candidate j not in current_code that covers all words in unique_mask
                        # We need (cover_matrix[j] & unique_mask) == unique_mask
                        # This is equivalent to np.sum(cover_matrix[j] & unique_mask) == np.sum(unique_mask)
                        # To speed this up, we can precompute the sum of unique_mask
                        target_sum = np.sum(unique_mask)
                        
                        # Iterate through potential candidates (all words)
                        # We can vectorize this: 
                        # intersections = cover_matrix & unique_mask[:, None]
                        # sums = np.sum(intersections, axis=1)
                        # This is O(N^2) which is ~4.7M ops. Fast in numpy.
                        
                        intersections = cover_matrix_uint8 & unique_mask[:, None].astype(np.uint8)
                        sums = intersections.sum(axis=1)
                        
                        # Candidates are those with sum == target_sum and not in code_set
                        # We need to exclude indices in code_set
                        # Create a mask of valid candidates
                        valid_mask = (sums == target_sum)
                        
                        # Exclude existing codewords
                        for c in code_set:
                            valid_mask[c] = False

                        candidate_indices = np.where(valid_mask)[0]
                        
                        if len(candidate_indices) > 0:
                            # Choose a random candidate
                            new_idx = candidate_indices[random.randint(0, len(candidate_indices) - 1)]
                            new_idx = int(new_idx)
                            
                            # Update cover_count
                            cover_count -= cover_matrix_int8[idx]
                            cover_count += cover_matrix_int8[new_idx]
                            
                            code_set.remove(idx)
                            code_set.add(new_idx)
                            current_code[i] = new_idx
                            improved = True
                            break
        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code, max_time=deadline - 10.0)
    
    # Run a few restarts if time permits
    if time.time() < deadline - 15.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code, max_time=deadline - 2.0)
            if len(cand_code) < len(best_code):
                best_code = cand_code
                # If we found a better solution, try to improve it further
                if time.time() < deadline - 5.0:
                    best_code = local_search_reduce(best_code, max_time=deadline - 2.0)

    # Convert indices back to words
    return [all_words[i] for i in best_code]
