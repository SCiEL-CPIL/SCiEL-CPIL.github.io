import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total_space = q ** n
    
    # Precompute index to word
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w
    
    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute powers of q for index arithmetic
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    
    # Build cover matrix efficiently using numpy
    # cover_matrix[i, j] = True if word j is within Hamming distance <= radius of word i
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    
    # Diagonal: every word covers itself
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        # For radius 1: for each position, for each value change
        # We can vectorize this
        for pos in range(n):
            power = powers[pos]
            # For each word i, the neighbors at position pos are:
            # i + (val - orig_val) * power for val != orig_val
            # We can compute this in a loop but it's O(total_space * n * (q-1))
            for i in range(total_space):
                orig_val = all_words[i][pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True
    
    # Convert to int8 for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Precompute neighbor lists for each codeword (indices of words it covers)
    # This helps with cover_count updates
    neighbor_lists = []
    for i in range(total_space):
        neighbors = np.where(cover_matrix[i])[0]
        neighbor_lists.append(neighbors)
    
    # Greedy Set Cover
    def greedy_cover():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int8 = uncovered.astype(np.int8)
        
        while np.any(uncovered):
            # Gain for each codeword = number of uncovered words it covers
            gains = cm_int8.dot(unc_int8)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking for diversity
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            unc_int8 = uncovered.astype(np.int8)
        
        return code
    
    # Local search: try to remove or swap codewords
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        n_cw = len(current_code)
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count[neighbor_lists[idx]] += 1
        
        improved = True
        max_iterations = 200  # Safety limit
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                neighbors = neighbor_lists[idx]
                
                # Check if removable: all covered words have count > 1
                if np.min(cover_count[neighbors]) > 1:
                    cover_count[neighbors] -= 1
                    current_code.pop(i)
                    improved = True
                    break
                
                # Try swap: find a replacement that covers all unique words
                unique_mask = mask & (cover_count == 1)
                
                if np.any(unique_mask):
                    unique_indices = np.where(unique_mask)[0]
                    
                    # Find candidates that cover all unique words
                    # A candidate c is valid if cover_matrix[c, u] is True for all u in unique_indices
                    # AND c is not already in current_code
                    current_set = set(current_code)
                    
                    # Check each potential replacement
                    # For efficiency, only check words that cover at least one unique word
                    # and are not in the current code
                    potential_replacements = []
                    for u in unique_indices:
                        # Words that cover u
                        covering_words = np.where(cover_matrix[:, u])[0]
                        for cw in covering_words:
                            if cw not in current_set and cw != idx:
                                potential_replacements.append(cw)
                    
                    if potential_replacements:
                        # Remove duplicates
                        potential_replacements = list(set(potential_replacements))
                        
                        # Find a replacement that covers all unique words
                        for new_idx in random.sample(potential_replacements, min(len(potential_replacements), 50)):
                            if all(cover_matrix[new_idx, u] for u in unique_indices):
                                # Perform swap
                                cover_count[neighbors] -= 1
                                new_neighbors = neighbor_lists[new_idx]
                                cover_count[new_neighbors] += 1
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                        
                        if improved:
                            break
        
        return current_code
    
    # 2-exchange local search: try swapping pairs of codewords
    def two_exchange_search(code):
        if not code or len(code) < 2:
            return code
        
        current_code = code[:]
        n_cw = len(current_code)
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count[neighbor_lists[idx]] += 1
        
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx1 = current_code[i]
                neighbors1 = neighbor_lists[idx1]
                
                # Words uniquely covered by idx1
                unique1 = cover_matrix[idx1] & (cover_count == 1)
                
                if not np.any(unique1):
                    # Can remove
                    cover_count[neighbors1] -= 1
                    current_code.pop(i)
                    improved = True
                    break
                
                for j in range(i + 1, n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx2 = current_code[j]
                    neighbors2 = neighbor_lists[idx2]
                    
                    # Words uniquely covered by idx2
                    unique2 = cover_matrix[idx2] & (cover_count == 1)
                    
                    # Try removing both idx1 and idx2
                    # After removal, what becomes uncovered?
                    # Words that had count 1 from idx1 or idx2 (and not from both)
                    
                    # Count after removing both
                    # For words covered by idx1: new count = old count - 1 (from idx1) - 1 (if also covered by idx2)
                    # For words covered by idx2: new count = old count - 1 (from idx2) - 1 (if also covered by idx1)
                    
                    # Simpler: compute the intersection
                    both_cover = cover_matrix[idx1] & cover_matrix[idx2]
                    
                    # Words that become uncovered after removing both:
                    # - Words covered only by idx1 (not by idx2, and count was 1)
                    # - Words covered only by idx2 (not by idx1, and count was 1)
                    # - Words covered by both, but count was 2 (so removing both makes it 0)
                    
                    only1 = cover_matrix[idx1] & ~cover_matrix[idx2]
                    only2 = cover_matrix[idx2] & ~cover_matrix[idx1]
                    
                    newly_uncovered = (only1 & (cover_count == 1)) | (only2 & (cover_count == 1)) | (both_cover & (cover_count == 2))
                    
                    if not np.any(newly_uncovered):
                        # Safe to remove both? No, we need to find replacements
                        # For now, just try removing one if possible
                        pass
                    
                    # Alternative: try swapping idx1 with some other word
                    # This is getting complex, so let's just do single removals/swaps
                    break
        
        return current_code
    
    # Main search loop
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)
    
    # Try multiple restarts with different random seeds
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation: start from greedy but with random choices
        random.seed(r * 12345 + 67890)
        cand_code = greedy_cover()
        cand_code = local_search_reduce(cand_code)
        
        if len(cand_code) < best_size:
            best_code = cand_code
            best_size = len(best_code)
        
        # Further improve with another local search pass
        if time.time() < deadline - 5.0:
            improved = local_search_reduce(best_code)
            if len(improved) < best_size:
                best_code = improved
                best_size = len(best_code)
    
    # Final polish: try to remove any remaining redundant codewords
    if time.time() < deadline - 2.0:
        final_code = local_search_reduce(best_code)
        best_code = final_code
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
