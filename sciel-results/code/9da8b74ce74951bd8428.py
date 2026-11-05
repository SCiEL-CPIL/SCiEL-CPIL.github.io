import itertools
import time
import random
import numpy as np

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

    # Precompute balls using numpy for speed
    # balls[c] = list of indices covered by codeword c
    # Use vectorized approach: for each position and each value change
    balls = []
    for c in range(total_words):
        covered = {c}
        # For each position
        for pos in range(n):
            # Current value at this position
            current_val = (c // (q ** pos)) % q
            # For each other value
            for new_val in range(q):
                if new_val != current_val:
                    delta = (new_val - current_val) * (q ** pos)
                    new_c = c + delta
                    covered.add(new_c)
        balls.append(list(covered))
    
    # Convert to numpy arrays for faster operations
    balls_arr = np.array(balls, dtype=np.int32)
    
    # For fast subset check, use a 2D boolean matrix: cover_matrix[c, w] = True if w is in balls[c]
    cover_matrix = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        cover_matrix[c, balls_arr[c]] = True

    # Initial code: Direct sum of ternary tetracode and full space of length 2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_code.append(word_to_idx(w))
            
    # Ensure distinct
    initial_code = list(set(initial_code))
    
    # Current code as a set for O(1) lookup
    code = set(initial_code)
    
    # Compute coverage counts using numpy
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        counts[balls_arr[c]] += 1

    # Local Search
    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    best_code_size = len(code)
    best_code_set = set(code)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Try to remove: find codeword with no unique words
        removable = False
        for c in list(code):
            # Check if c has any unique words
            unique_mask = (counts == 1) & cover_matrix[c]
            if not np.any(unique_mask):
                # Remove c
                code.remove(c)
                counts[balls_arr[c]] -= 1
                improved = True
                removable = True
                break
        
        if improved:
            no_improve_count = 0
            if len(code) < best_code_size:
                best_code_size = len(code)
                best_code_set = set(code)
            continue
        
        # Try to swap
        swap_done = False
        # Get codewords sorted by number of unique words (fewer unique = easier to swap)
        code_list = list(code)
        unique_counts = np.array([np.sum((counts == 1) & cover_matrix[c]) for c in code_list])
        sorted_indices = np.argsort(unique_counts)
        
        for idx in sorted_indices:
            c = code_list[idx]
            unique_count = unique_counts[idx]
            if unique_count == 0:
                continue
            if unique_count > 5:  # Skip if too many unique words to swap
                continue
                
            # Find unique words
            unique_words = np.where((counts == 1) & cover_matrix[c])[0]
            
            # Find candidate c_prime that covers all unique words
            # c_prime must not be in code
            found_swap = False
            # Precompute which candidates cover all unique words
            # For each candidate, check if it covers all unique_words
            candidates = np.setdiff1d(np.arange(total_words), np.array(code_list))
            
            for c_prime in candidates:
                # Check if balls[c_prime] covers all words in unique_words
                if np.all(cover_matrix[c_prime, unique_words]):
                    # Perform swap
                    code.remove(c)
                    code.add(c_prime)
                    counts[balls_arr[c]] -= 1
                    counts[balls_arr[c_prime]] += 1
                    found_swap = True
                    break
            
            if found_swap:
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
                if len(code) > 71:  # Only kick if we can potentially improve
                    # Add 2 random codewords not in code
                    code_list = list(code)
                    candidates = np.setdiff1d(np.arange(total_words), np.array(code_list))
                    
                    if len(candidates) >= 2:
                        kick_indices = random.sample(range(len(candidates)), 2)
                        for ki in kick_indices:
                            c_new = candidates[ki]
                            code.add(c_new)
                            counts[balls_arr[c_new]] += 1
                        
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
                    candidates = np.setdiff1d(np.arange(total_words), np.array(code_list))
                    
                    if len(candidates) >= 1:
                        kick_idx = random.randint(0, len(candidates) - 1)
                        c_new = candidates[kick_idx]
                        code.add(c_new)
                        counts[balls_arr[c_new]] += 1
                        
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
