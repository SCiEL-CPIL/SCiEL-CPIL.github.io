import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    total_words = q ** n
    
    # Map tuple to index
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(w):
            idx += val * (q ** i)
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute balls using vectorized approach
    # For radius 1, a ball contains the center and n*(q-1) neighbors
    ball_size = 1 + n * (q - 1)
    
    # Create an array to hold all ball contents
    # Shape: (total_words, ball_size)
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    
    # Center is the index itself
    balls[:, 0] = np.arange(total_words)
    
    # Fill neighbors
    col_idx = 1
    for pos in range(n):
        power = q ** pos
        for new_val in range(q):
            # We need to iterate over all words? No, we can do this smarter.
            # For a word with index c, the value at pos is (c // power) % q.
            # If new_val != current_val, the neighbor is c + (new_val - current_val) * power.
            # This is hard to vectorize directly without knowing current_val for all c at once.
            # Let's extract current values for all words at this position.
            current_vals = (np.arange(total_words) // power) % q
            # For this specific new_val, we only care about words where current_vals != new_val
            mask = (current_vals != new_val)
            if np.any(mask):
                # Calculate deltas for those words
                # delta = (new_val - current_vals[mask]) * power
                deltas = (new_val - current_vals[mask]) * power
                # The neighbors are c + delta
                neighbors = np.arange(total_words)[mask] + deltas
                balls[mask, col_idx] = neighbors
            col_idx += 1
            
    # Build cover matrix
    # cover_matrix[c, w] is True if w is in balls[c]
    cover_matrix = np.zeros((total_words, total_words), dtype=bool)
    # Use advanced indexing
    # rows: [0, 0, ..., 0, 1, 1, ..., 1, ..., total_words-1, ...]
    rows = np.repeat(np.arange(total_words), ball_size)
    cols = balls.ravel()
    cover_matrix[rows, cols] = True

    # Initial code: Direct sum of ternary tetracode and full space of length 2
    # Tetracode is a perfect 1-error correcting code of length 4 over GF(3)
    # It has 9 words.
    # Product with 3^(n-4) gives 9 * 3^(n-4) words.
    # For n=6, this is 9 * 9 = 81 words.
    # We want to reduce this to 73 or lower.
    
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                initial_code.append(word_to_idx(w))
    else:
        # Fallback for small n (not expected in this task, but safe)
        initial_code = list(range(total_words))

    # Ensure distinct
    initial_code = list(set(initial_code))
    
    # Current code as a set for O(1) lookup
    code = set(initial_code)
    
    # Compute coverage counts
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        counts[balls[c]] += 1

    # Local Search
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_code_size = len(code)
    best_code_set = set(code)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove: find codeword with no unique words
        # A codeword c is removable if all words in its ball are covered by at least one other codeword.
        # This is equivalent to: for all w in ball(c), count(w) > 1.
        # Or: there is no w in ball(c) such that count(w) == 1.
        
        # Optimization: Only check codewords that *might* be removable.
        # If a codeword has many unique words, it's not removable.
        # We can precompute unique_counts for all codewords in the code.
        
        code_list = list(code)
        if not code_list:
            break
            
        # Vectorized check for unique words per codeword in code
        # unique_mask[c, w] = (counts[w] == 1) and cover_matrix[c, w]
        # We want sum(unique_mask[c, :]) == 0 for some c in code.
        
        # Instead of checking all 729 words for each c, we can iterate.
        # But let's try to vectorize the "is removable" check for all c in code.
        
        # Get indices of current code
        code_arr = np.array(code_list, dtype=np.int32)
        
        # For each c in code_arr, check if any word in its ball has count 1
        # unique_counts_for_code = np.array([np.sum((counts == 1) & cover_matrix[c]) for c in code_arr])
        # This is a loop over code size (max ~80). 80 * 729 operations is ~58k, very fast.
        
        unique_counts = np.array([np.sum((counts == 1) & cover_matrix[c]) for c in code_arr])
        
        # Find indices in code_arr where unique_counts == 0
        removable_indices = np.where(unique_counts == 0)[0]
        
        if len(removable_indices) > 0:
            # Pick the first one (or random)
            idx_to_remove = removable_indices[0]
            c = code_arr[idx_to_remove]
            
            code.remove(c)
            counts[balls[c]] -= 1
            improved = True
            no_improve_count = 0
            if len(code) < best_code_size:
                best_code_size = len(code)
                best_code_set = set(code)
            continue
        
        # 2. Try to swap
        # Sort codewords by unique_counts (ascending) to try swapping "weak" codewords first
        sorted_indices = np.argsort(unique_counts)
        
        swap_done = False
        for idx in sorted_indices:
            c = code_arr[idx]
            u_count = unique_counts[idx]
            
            if u_count == 0:
                # Should have been removed, but maybe counts changed?
                # Or if we didn't remove in the previous step because we broke?
                # We continue if 0.
                continue
                
            if u_count > 10:  # Threshold: if too many unique words, swapping is hard
                continue
                
            # Find the unique words
            unique_words = np.where((counts == 1) & cover_matrix[c])[0]
            
            # Find candidates c' that cover all unique_words
            # Condition: cover_matrix[c', unique_words] must be all True.
            # This is equivalent to: np.all(cover_matrix[:, unique_words], axis=1)
            
            # Note: c' must not be in the current code (unless we allow self-swap which is useless)
            # And c' should ideally be "better" or at least not worse.
            # For now, any valid swap is an improvement in terms of diversity, 
            # but we need to ensure the resulting code is still a cover.
            # If c' covers all unique words of c, and we remove c and add c',
            # do we lose coverage of any other words?
            # The words covered by c are B_c.
            # Unique words U_c are covered only by c.
            # If c' covers U_c, then after swap, U_c is covered by c'.
            # What about non-unique words in B_c? They are covered by others, so removing c is safe for them.
            # What about words covered by c' but not c? They are just newly covered (or re-covered).
            # So the swap is valid if U_c subseteq B_{c'}.
            
            # Vectorized check for all c' in space
            # candidate_mask[c_prime] = True if c_prime covers all unique_words
            if len(unique_words) == 0:
                continue
                
            # Check all 729 words
            # cover_matrix[:, unique_words] has shape (729, len(unique_words))
            # np.all(..., axis=1) has shape (729,)
            candidate_mask = np.all(cover_matrix[:, unique_words], axis=1)
            
            # Exclude current code words (except c itself, but we're replacing c, so c is removed)
            # We want c' not in code - {c}
            # Actually, if c' is in code, swapping c with c' just removes c (if c' already there) or is a no-op.
            # We want to add a NEW word or replace with a DIFFERENT existing word?
            # If we replace c with c' where c' is already in code, we just remove c.
            # That's a removal, which we already tried.
            # So we should look for c' NOT in code.
            
            # Create a mask of words NOT in code
            not_in_code_mask = np.ones(total_words, dtype=bool)
            not_in_code_mask[code_arr] = False
            
            # Valid candidates: candidate_mask AND not_in_code_mask
            valid_candidates = candidate_mask & not_in_code_mask
            
            # Also, we might want to prefer candidates that cover many "low count" words to improve robustness?
            # For now, just pick the first one or a random one.
            valid_idx = np.where(valid_candidates)[0]
            
            if len(valid_idx) > 0:
                # Pick a random valid candidate to escape local minima
                c_prime = valid_idx[random.randint(0, len(valid_idx) - 1)]
                
                # Perform swap
                code.remove(c)
                code.add(c_prime)
                counts[balls[c]] -= 1
                counts[balls[c_prime]] += 1
                
                improved = True
                swap_done = True
                no_improve_count = 0
                if len(code) < best_code_size:
                    best_code_size = len(code)
                    best_code_set = set(code)
                break
                
        if not improved:
            no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                # Try a kick
                # Add 2 random codewords not in code to escape local minimum
                if len(code) > 71:  # Only kick if we can potentially improve
                    code_list = list(code)
                    # Find words not in code
                    not_in_code_mask = np.ones(total_words, dtype=bool)
                    not_in_code_mask[code_arr] = False
                    candidates = np.where(not_in_code_mask)[0]
                    
                    if len(candidates) >= 2:
                        kick_indices = random.sample(list(candidates), 2)
                        for ki in kick_indices:
                            c_new = ki
                            code.add(c_new)
                            counts[balls[c_new]] += 1
                        
                        no_improve_count = 0
                        improved = True  # Force another iteration
                    else:
                        break
                else:
                    break
            elif no_improve_count >= max_no_improve // 2:
                # Try a single kick
                if len(code) > 71:
                    code_list = list(code)
                    not_in_code_mask = np.ones(total_words, dtype=bool)
                    not_in_code_mask[code_arr] = False
                    candidates = np.where(not_in_code_mask)[0]
                    
                    if len(candidates) >= 1:
                        kick_idx = random.randint(0, len(candidates) - 1)
                        c_new = candidates[kick_idx]
                        code.add(c_new)
                        counts[balls[c_new]] += 1
                        
                        no_improve_count = 0
                        improved = True  # Force another iteration
                    else:
                        break
                else:
                    break

    # Use the best code found
    code = best_code_set
    
    # Return the list of codewords (as lists of integers)
    return [idx_to_word(c) for c in code]
