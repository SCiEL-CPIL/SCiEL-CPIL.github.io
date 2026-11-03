import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Total words
    total_words = q ** n
    
    # Precompute neighbors and masks using NumPy for speed
    # cover_matrix[i, j] is True if j is within radius of i
    # Here radius is 1, so j is i or differs in 1 position.
    
    # Generate all words
    # Using itertools.product is slow for large N, but for 2187 it's okay.
    # We can generate indices directly.
    indices = np.arange(total_words)
    
    # Convert indices to words (n-d array of digits)
    # We need a fast way to check distance.
    # Precomputing the full cover matrix (2187x2187) is ~4.8M entries.
    # As bool, that's ~5MB. Very manageable.
    
    # To build cover_matrix efficiently:
    # For each word i, its neighbors are i itself and words differing in 1 digit.
    # We can generate the "step" vectors.
    
    # Let's just loop over i and compute neighbors, filling the matrix.
    # This is O(N * N * n) if naive, but we can be smarter.
    # Actually, for N=2187, a simple loop in Python might be slow if not careful.
    # Let's use the property of base-q representation.
    
    cover_matrix = np.zeros((total_words, total_words), dtype=bool)
    
    # Precompute digit values for all indices
    # digits[i, j] is the j-th digit of index i
    digits = np.zeros((total_words, n), dtype=np.int32)
    for pos in range(n):
        # Extract digit at pos
        # idx = d_0 * q^(n-1) + ... + d_pos * q^(n-1-pos) + ...
        # Actually standard base conversion:
        # d_i = (idx // q^(n-1-i)) % q
        # Let's just compute it once.
        divs = q ** (n - 1 - pos)
        digits[:, pos] = (indices // divs) % q
        
    # Now for each word i, find neighbors
    # Neighbor j differs from i in at most 1 position.
    # We can iterate through all pairs? No, too slow.
    # For each i, for each pos, for each new_val != digit:
    # Calculate new index.
    
    # This loop is N * n * (q-1) = 2187 * 7 * 2 ~ 30k iterations.
    # Inside, we do array operations? No, just index calculation.
    # This part is fast enough in Python.
    
    for i in range(total_words):
        # Self
        cover_matrix[i, i] = True
        
        # Neighbors
        # We need to map digit changes to index changes.
        # If we change digit at pos from d to d', index changes by (d'-d) * q^(n-1-pos)
        divs = q ** (n - 1 - pos) # Wait, pos is 0-indexed from left?
        # My digits array: digits[i, pos] is digit at position pos (0 is most significant)
        # So weight is q^(n-1-pos)
        
        for pos in range(n):
            orig_val = digits[i, pos]
            weight = q ** (n - 1 - pos)
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = new_val - orig_val
                new_idx = i + diff * weight
                cover_matrix[i, new_idx] = True

    # Now we have cover_matrix.
    # best_solution
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        while not np.all(covered):
            # Gain for each candidate c is number of new words covered
            # gain[c] = sum(cover_matrix[c] & ~covered)
            # Vectorized:
            # ~covered is a bool array.
            # cover_matrix & ~covered is a bool matrix.
            # Sum along axis 1.
            gains = np.sum(cover_matrix & ~covered, axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Should not happen if not all covered
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            chosen = candidates[np.random.randint(len(candidates))]
            code.append(chosen)
            covered |= cover_matrix[chosen]
            
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        if not code:
            return code
            
        # cover_counts[w] = number of codewords covering w
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_counts += cover_matrix[c]
            
        # in_code
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
            
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            # We iterate backwards to handle pops easily? 
            # Or just mark for removal.
            to_remove = []
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any word covered only by c
                # unique_words = cover_matrix[c] & (cover_counts == 1)
                # If any, c is not redundant.
                # If none, c is redundant.
                # Note: cover_counts includes c.
                # So if cover_counts[w] == 1 and cover_matrix[c, w] is True, then w is only covered by c.
                # We want to check if there EXISTS such w.
                
                # Vectorized check:
                # masked_counts = np.where(cover_matrix[c], cover_counts, 0)
                # if np.any(masked_counts == 1): redundant = False
                # else: redundant = True
                
                # Simpler:
                # If min(cover_counts[cover_matrix[c]]) > 1, then redundant.
                # But we need to be careful with boolean indexing.
                
                # Let's use:
                # counts_for_c = cover_counts[cover_matrix[c]]
                # if len(counts_for_c) == 0: redundant (shouldn't happen)
                # if np.min(counts_for_c) > 1: redundant
                
                counts_for_c = cover_counts[cover_matrix[c]]
                if np.min(counts_for_c) > 1:
                    # Redundant
                    cover_counts -= cover_matrix[c]
                    in_code[c] = False
                    code.pop(i)
                    changed = True
                    if time.time() > deadline:
                        return code
        return code

    def local_search(code):
        if not code:
            return code
            
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_counts += cover_matrix[c]
            
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            # We can do this in a batch or one by one.
            # One by one is safer for update logic.
            removed_any = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                counts_for_c = cover_counts[cover_matrix[c]]
                if np.min(counts_for_c) > 1:
                    cover_counts -= cover_matrix[c]
                    in_code[c] = False
                    code.pop(i)
                    removed_any = True
                    improved = True
                    if time.time() > deadline:
                        return code
            if removed_any:
                continue # Restart loop to catch new redundancies

            # 2. Try swaps
            # For each c in code, find a c' not in code such that swapping reduces size or enables removal.
            # Actually, standard swap: replace c with c' if c' covers all "unique" words of c.
            # Unique words of c: w such that cover_counts[w] == 1 and cover_matrix[c, w] == True.
            
            for i in range(len(code)):
                c = code[i]
                # Find unique words covered by c
                # unique_mask = cover_matrix[c] & (cover_counts == 1)
                unique_mask = cover_matrix[c] & (cover_counts == 1)
                
                # If no unique words, it's redundant (should have been caught above, but check again just in case)
                if not np.any(unique_mask):
                    cover_counts -= cover_matrix[c]
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break # Restart loop
                
                # We need to find c' not in code such that cover_matrix[c'] covers unique_mask.
                # i.e., np.all(cover_matrix[c'][unique_mask]) is True.
                
                # To speed this up, we can pre-filter candidates?
                # Or just iterate. N=2187.
                # Checking one c' takes O(num_unique).
                # Total O(N_code * N_candidates * N_unique).
                # N_code ~ 180. N_candidates ~ 2000. N_unique ~ 10-20.
                # 180 * 2000 * 20 = 7.2M ops. In NumPy, this is fast if vectorized.
                
                # Vectorized check for all c' at once?
                # cover_matrix is (N, N).
                # unique_mask is (N,).
                # We want to know for each c' if cover_matrix[c', unique_mask] are all True.
                # subset = cover_matrix[:, unique_mask] # Shape (N, num_unique)
                # valid = np.all(subset, axis=1) # Shape (N,)
                
                # But we only care about c' not in code.
                # valid_candidates = valid & ~in_code
                
                # However, creating `subset` every iteration is expensive (memory copy).
                # Let's stick to a loop over candidates but vectorize the inner check?
                # Or loop over c' and check.
                
                # Let's try vectorizing the candidate selection.
                # We can precompute nothing here easily without heavy memory.
                # Let's just loop over potential c'.
                # Optimization: Only check c' that are NOT in code.
                
                found_swap = False
                # Iterate over all indices
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    
                    # Check if c_prime covers all unique words of c
                    # If unique_mask is empty, handled above.
                    # Check: np.all(cover_matrix[c_prime, unique_mask])
                    if np.all(cover_matrix[c_prime, unique_mask]):
                        # Perform swap
                        # Remove c
                        cover_counts -= cover_matrix[c]
                        in_code[c] = False
                        
                        # Add c_prime
                        cover_counts += cover_matrix[c_prime]
                        in_code[c_prime] = True
                        code[i] = c_prime
                        
                        found_swap = True
                        improved = True
                        break # Restart main while loop to check for new redundancies
                
                if found_swap:
                    break
                
                if time.time() > deadline:
                    return code
            
        return code

    # Main loop
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
        
    if best_solution is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        # Convert idx to word
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        result.append(word)
        
    return result
