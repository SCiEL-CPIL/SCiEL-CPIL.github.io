import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit, 32s buffer)

    total_space = q ** n
    nb_size = 1 + n * (q - 1)

    # Precompute neighbors as a numpy array for vectorized operations
    # neighbors[i] contains indices of all words within Hamming distance <= radius of i
    # For q=3, n=7, radius=1: 1 + 7*2 = 15 neighbors
    neighbors = np.zeros((total_space, nb_size), dtype=np.int32)
    
    # Efficient neighbor generation using base-q arithmetic
    # Precompute the "step" values for each position
    # step[pos] = q^pos (the value of changing digit at position pos by 1)
    # But we need to be careful with carries. Let's use a simpler approach.
    
    # For each word index, generate its neighbors by modifying each digit
    # We'll use a vectorized approach where possible, but for correctness and clarity,
    # we'll precompute using a loop but with optimized inner logic.
    
    # Precompute powers of q
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    def idx_to_digits(idx):
        digits = np.zeros(n, dtype=np.int32)
        tmp = idx
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        return digits

    def digits_to_idx(digits):
        idx = 0
        for pos in range(n):
            idx += digits[pos] * q_powers[pos]
        return idx

    for i in range(total_space):
        neighbors[i, 0] = i  # The word itself
        
        if radius >= 1:
            digits = idx_to_digits(i)
            nb_idx = 1
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = val
                    new_idx = digits_to_idx(new_digits)
                    neighbors[i, nb_idx] = new_idx
                    nb_idx += 1
    
    # Greedy Set Cover
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Calculate gains: for each candidate, how many new words it covers
            # gains[i] = number of neighbors of i that are not yet covered
            gains = np.sum(~covered[neighbors], axis=1)
            
            if randomize:
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 0:
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            code.append(int(best_idx))
            covered[neighbors[best_idx]] = True
        
        return code

    def local_search_reduce(code, time_budget=0.5):
        """Remove redundant codewords using coverage counts."""
        if not code:
            return code
        
        current_code = list(code)
        
        if len(current_code) == 0:
            return current_code
            
        # Build cover_count using bincount
        flat_neighbors = neighbors[current_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        improved = True
        t0 = time.time()
        while improved and (time.time() - t0) < time_budget:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                idx = current_code[i]
                
                # Check if all words covered by idx are covered by at least one other codeword
                min_count = np.min(cover_count[neighbors[idx]])
                
                if min_count > 1:
                    # Remove idx
                    cover_count[neighbors[idx]] -= 1
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    def local_search_swap(code, time_budget=1.0):
        """
        Try to swap codewords to reduce the set size.
        Strategy: Remove a codeword, find the best replacement that covers the most newly-uncovered words,
        then run reduction. Accept if the result is smaller.
        """
        if not code:
            return code
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        if best_len == 0:
            return best_code
            
        # Build cover_count
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        t0 = time.time()
        attempts = 0
        
        while (time.time() - t0) < time_budget:
            attempts += 1
            
            # Pick a random codeword to potentially remove
            i = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i]
            
            # Tentatively remove idx_out
            cover_count[neighbors[idx_out]] -= 1
            current_code.pop(i)
            
            # Find uncovered words (those with count 0)
            uncovered_mask = (cover_count == 0)
            
            if not np.any(uncovered_mask):
                # It was redundant; restore and continue (reduction should catch this)
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Find the best replacement: maximize coverage of uncovered words
            # gains[j] = number of neighbors of j that are currently uncovered
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude codewords already in current_code (to avoid no-op swaps)
            # But we can't easily vectorize exclusion of a dynamic list.
            # Instead, just pick the argmax. If it's already in the set, it's a no-op.
            best_j = int(np.argmax(gains))
            
            if gains[best_j] == 0:
                # No improvement possible with a single swap; restore
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Add best_j
            current_code.append(best_j)
            cover_count[neighbors[best_j]] += 1
            
            # Run a quick reduction pass
            new_code = local_search_reduce(current_code, time_budget=0.1)
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                # Rebuild cover_count for the new best_code
                flat_neighbors = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_neighbors, minlength=total_space)
                current_code = best_code[:]
            else:
                # Revert: remove best_j, restore idx_out
                # This is tricky because local_search_reduce may have modified the set.
                # We need to restore to the state before the swap.
                # Since we don't have a clean snapshot, let's rebuild from best_code
                # and try again.
                current_code = best_code[:]
                flat_neighbors = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        return best_code

    # Strategy:
    # 1. Run deterministic greedy + reduce + swap.
    # 2. Run randomized greedy + reduce + swap multiple times.
    # 3. Keep the best.

    # Initial deterministic run
    best_code = greedy_cover(randomize=False)
    best_code = local_search_reduce(best_code, time_budget=1.0)
    best_code = local_search_swap(best_code, time_budget=2.0)
    best_len = len(best_code)
    
    # Randomized restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        cand_code = greedy_cover(randomize=True)
        cand_code = local_search_reduce(cand_code, time_budget=0.5)
        cand_code = local_search_swap(cand_code, time_budget=1.0)
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code
            
    # Convert indices back to words
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code]
