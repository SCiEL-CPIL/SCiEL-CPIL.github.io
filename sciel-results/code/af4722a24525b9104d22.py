import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
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

    # Precompute neighbors as a numpy array for vectorized operations
    # cover_count is the size of the Hamming ball of radius 1
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # Efficiently generate neighbors
    # For q=3, n=7, radius=1:
    # Center: (w0, w1, ...)
    # Neighbors: change one position to one of q-1 other values.
    # We can compute indices directly using modular arithmetic or lookup.
    # Given total_words is small (2187), a simple loop is fine for precomputation.
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        # Ensure length is exactly cover_count
        if len(idxs) < cover_count:
             # Should not happen, but pad just in case
             idxs += [i] * (cover_count - len(idxs))
        neighbors_arr[i, :] = idxs[:cover_count]

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation: for each candidate c, how many uncovered words does it cover?
            # neighbors_arr[c] gives the words covered by c.
            # We want sum(uncovered[neighbors_arr[c]])
            # This is a matrix-vector product if uncovered is treated as a vector.
            # But neighbors_arr is 2D. 
            # gains = np.sum(uncovered[neighbors_arr], axis=1) is not quite right syntax for fancy indexing sum.
            # Correct way:
            # Create a matrix where rows are candidates, cols are their neighbors.
            # uncovered is a vector of length total_words.
            # We need to gather uncovered values for each candidate's neighbors.
            
            # Optimization:
            # gains = np.zeros(total_words, dtype=np.int32)
            # for c in range(total_words):
            #    gains[c] = np.sum(uncovered[neighbors_arr[c]])
            # This is O(N^2 * R) which might be slow in Python loop.
            
            # Better: Use matrix multiplication?
            # Let M be the incidence matrix (total_words x total_words) where M[i,j]=1 if j is neighbor of i.
            # Then gains = M @ uncovered_vector.
            
            if not hasattr(greedy_construct_vectorized, 'incidence_matrix'):
                # Build incidence matrix once
                inc = np.zeros((total_words, total_words), dtype=np.float32)
                for i in range(total_words):
                    inc[i, neighbors_arr[i]] = 1.0
                greedy_construct_vectorized.incidence_matrix = inc
            
            uncovered_vec = uncovered.astype(np.float32)
            gains = greedy_construct_vectorized.incidence_matrix @ uncovered_vec
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
        """Deterministically remove redundant codewords."""
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                # If any word count becomes 0, this codeword is necessary
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def simulated_annealing(code, deadline):
        """
        Structural change: Simulated Annealing to escape local minima.
        Moves:
        1. Remove redundant codeword (if exists). Always accept.
        2. Swap codeword i with random j. Accept if size same (neutral) or smaller.
           To escape minima, we accept neutral swaps with prob 1.0 initially, cooling down.
           Actually, for covering codes, size is the only cost. 
           If we swap and size stays same, it's a neutral move.
           If we add a codeword, size increases. We accept with prob exp(-beta * delta_size).
           If we remove a non-redundant codeword, we break validity. We don't do that.
        
        Strategy:
        - Maintain a valid code.
        - Try to remove redundant words.
        - Try swaps (1-for-1). If the swap keeps the code valid (all words covered), it's a neutral move (size unchanged).
          We accept neutral moves to random-walk the space of minimal solutions.
        - Occasionally, try to ADD a word (size +1) to "jump" to a different region, then try to remove words.
          We accept adds with low probability.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        # Temperature schedule
        T = 1.0
        cooling_rate = 0.99
        min_T = 0.01
        
        # Precompute list of all possible codewords for random sampling
        all_codewords = np.arange(total_words)
        
        while time.time() < deadline and T > min_T:
            improved = False
            
            # 1. Try to remove a redundant codeword
            # A codeword c is redundant if removing it doesn't make any count 0.
            # We can check this by iterating through code.
            # For speed, we pick a random subset to check.
            
            # Actually, let's just do a full pass for removal first if T is high, or random?
            # Let's try random removals.
            
            for _ in range(10): # Try a few random moves per temperature step
                if time.time() >= deadline:
                    return code
                
                move_type = random.random()
                
                if move_type < 0.3:
                    # Try Remove
                    if len(code) > 0:
                        idx = random.randint(0, len(code) - 1)
                        c = code[idx]
                        counts[neighbors_arr[c, :]] -= 1
                        if np.any(counts == 0):
                            # Revert
                            counts[neighbors_arr[c, :]] += 1
                        else:
                            # Accept removal
                            code.pop(idx)
                            improved = True
                            # Restart loop or continue? Continue to try more moves.
                            # But counts changed. We need to be careful.
                            # Since we popped, indices shift.
                            # Let's break and restart the inner loop logic?
                            # Or just continue. The counts are correct for the new code.
                            # But the `code` list changed.
                            # It's safer to break and re-evaluate in next iteration of while loop?
                            # No, we can just continue.
                            pass
                            
                elif move_type < 0.8:
                    # Try Swap
                    if len(code) > 0:
                        idx = random.randint(0, len(code) - 1)
                        c_old = code[idx]
                        c_new = random.choice(all_codewords.tolist())
                        
                        if c_old == c_new:
                            continue
                        
                        # Check if swapping c_old with c_new keeps code valid.
                        # We need to ensure all words covered by c_old are also covered by c_new 
                        # OR by other codewords.
                        # i.e., for all w in neighbors(c_old), counts[w] - 1 (removing c_old) + 1 (adding c_new) > 0?
                        # Actually, we remove c_old first.
                        counts[neighbors_arr[c_old, :]] -= 1
                        
                        # Check if any word is now uncovered (count == 0)
                        # If so, c_new MUST cover it.
                        # Let U be set of words with count == 0.
                        # We need neighbors(c_new) to include all w in U.
                        
                        uncovered_mask = (counts == 0)
                        if np.any(uncovered_mask):
                            # c_new must cover all these words
                            # Check if any word in U is NOT in neighbors(c_new)
                            # neighbors(c_new) is a list of indices.
                            # We can check: for each w in U, is w in neighbors(c_new)?
                            # Vectorized check:
                            # Create a boolean array for neighbors of c_new
                            # neigh_c_new = np.zeros(total_words, dtype=bool)
                            # neigh_c_new[neighbors_arr[c_new]] = True
                            # If any(w in U and not neigh_c_new[w]) -> Invalid.
                            
                            # Optimization:
                            # Count how many of the uncovered words are covered by c_new.
                            # It must be ALL of them.
                            # num_uncovered = np.sum(uncovered_mask)
                            # covered_by_new = np.sum(uncovered_mask[neighbors_arr[c_new]])
                            # if covered_by_new < num_uncovered: Invalid.
                            
                            num_unc = np.sum(uncovered_mask)
                            # neighbors_arr[c_new] gives the indices.
                            # uncovered_mask is a boolean array.
                            # uncovered_mask[neighbors_arr[c_new]] gives boolean values for those neighbors.
                            covered_by_new = np.sum(uncovered_mask[neighbors_arr[c_new, :]])
                            
                            if covered_by_new < num_unc:
                                # Revert
                                counts[neighbors_arr[c_old, :]] += 1
                                continue
                            
                            # Valid swap
                            # Apply c_new
                            counts[neighbors_arr[c_new, :]] += 1
                            code[idx] = c_new
                            # Size unchanged, so neutral move.
                            # Accept with probability 1 (or based on T if we had secondary cost, but here just accept to random walk)
                            # We always accept valid neutral moves in this simple scheme to explore.
                            improved = True
                        else:
                            # No words were uncovered by removing c_old.
                            # So any c_new is valid?
                            # Yes, because all words were covered by >= 2 codewords (or 1, but removing c_old didn't make it 0, so it was >=2? 
                            # Wait. If count was 1, removing makes it 0. 
                            # If count was >= 2, removing makes it >= 1.
                            # So if no count is 0, all words are still covered.
                            # So any c_new is a valid replacement.
                            counts[neighbors_arr[c_new, :]] += 1
                            code[idx] = c_new
                            improved = True

                else:
                    # Try Add (Expansion)
                    # This allows escaping local minima where no swap works.
                    # We add a random codeword that covers many "weakly covered" words?
                    # Or just random.
                    # Let's add a random codeword.
                    c_new = random.choice(all_codewords.tolist())
                    
                    # Check if c_new is already in code.
                    if c_new in code:
                        continue
                    
                    # Calculate "gain" or just accept with low prob?
                    # In SA, we accept "worse" moves (larger size) with prob exp(-delta/T).
                    # delta = 1 (size increases by 1).
                    prob = np.exp(-1.0 / T)
                    
                    if random.random() < prob:
                        code.append(c_new)
                        counts[neighbors_arr[c_new, :]] += 1
                        improved = True
            
            T *= cooling_rate
            
            # If we improved (removed a word), we might want to cool down slower?
            # Standard SA: T always cools.
            
        return code

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 5:
        # 1. Greedy Initialization
        code = greedy_construct_vectorized(seed)
        
        # 2. Deterministic Reduction
        code = reduce_code_fast(code)
        
        current_size = len(code)
        
        # 3. Simulated Annealing
        if time.time() < deadline - 10:
            code = simulated_annealing(code, deadline - 5)
            code = reduce_code_fast(code)
            
            if len(code) < current_size:
                current_size = len(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        seed += 1
        # Increase restart frequency if we are finding good solutions quickly?
        # Or just keep going.
        
    if best_code is None:
        # Fallback
        best_code = []
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
