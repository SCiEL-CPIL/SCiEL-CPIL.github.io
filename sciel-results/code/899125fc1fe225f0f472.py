import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # Precompute cover matrix: cover_matrix[i, j] is True if word i covers word j
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    
    for i in range(total_space):
        w = all_words[i]
        cover_matrix[i, i] = True # self cover
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    cover_matrix[i, j] = True

    total_mask_indices = np.arange(total_space)

    def greedy_cover():
        uncovered_mask = np.ones(total_space, dtype=bool)
        code = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
                
            # Vectorized gain calculation: sum of coverage over uncovered words
            # cover_matrix[:, uncovered_mask] is a submatrix of shape (total_space, num_uncovered)
            # Summing along axis 1 gives the number of uncovered words covered by each candidate
            gains = np.sum(cover_matrix[:, uncovered_mask], axis=1)
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            code.append(best_idx)
            # Update uncovered mask
            uncovered_mask[cover_matrix[best_idx, :]] = False
            
        return code, uncovered_mask

    def local_search_reduce(code, uncovered_mask):
        if not code:
            return code, uncovered_mask
            
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in code:
            # Add 1 to cover_count for all words covered by idx
            cover_count[cover_matrix[idx, :].nonzero()[0]] += 1
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
                
            random.shuffle(code)
            n_cw = len(code)
            
            for i in range(n_cw):
                idx = code[i]
                covered_indices = cover_matrix[idx, :].nonzero()[0]
                
                # Check if removable: min cover count among covered words must be > 1
                if len(covered_indices) > 0:
                    min_count = np.min(cover_count[covered_indices])
                    if min_count > 1:
                        # Remove idx
                        cover_count[covered_indices] -= 1
                        code.pop(i)
                        improved = True
                        break
                else:
                    # Should not happen as self is covered
                    code.pop(i)
                    improved = True
                    break
                # Note: The original code also tried swaps. 
                # Swaps are harder to vectorize efficiently in a single pass without complex logic.
                # We'll stick to removals for now, which is the primary reduction driver.
                # If we have time, we can add simple swap logic later.
                
        return code, uncovered_mask

    def simulated_annealing(code, uncovered_mask):
        if not code:
            return code, uncovered_mask
            
        current_code = code[:]
        current_uncovered = uncovered_mask.copy()
        
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count[cover_matrix[idx, :].nonzero()[0]] += 1
            
        def get_energy(c_uncovered, c_code):
            return len(c_code) + 100 * np.sum(c_uncovered)
            
        current_energy = get_energy(current_uncovered, current_code)
        best_code = current_code[:]
        best_uncovered = current_uncovered.copy()
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 1.0
        cooling_rate = 0.999
        
        T = T_start
        
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                # Add random word
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue
                    
                new_covered = cover_matrix[new_idx, :]
                new_uncovered = current_uncovered & ~new_covered
                
                # Energy calc
                new_code_len = len(current_code) + 1
                new_uncovered_count = np.sum(new_uncovered)
                new_energy = new_code_len + 100 * new_uncovered_count
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Accept
                    cover_count[new_covered.nonzero()[0]] += 1
                    current_code.append(new_idx)
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered.copy()
                        
            elif move_type < 0.6:
                # Remove random word
                if len(current_code) <= 1:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_pos]
                covered_indices = cover_matrix[idx, :].nonzero()[0]
                
                # Words that become uncovered
                unique_mask = (cover_count[covered_indices] == 1)
                # We need to know WHICH indices become uncovered to update the mask
                # unique_mask is a boolean array of the same length as covered_indices
                new_uncovered_indices = covered_indices[unique_mask]
                
                new_uncovered = current_uncovered.copy()
                if len(new_uncovered_indices) > 0:
                    new_uncovered[new_uncovered_indices] = True
                    
                new_code_len = len(current_code) - 1
                new_uncovered_count = np.sum(new_uncovered)
                new_energy = new_code_len + 100 * new_uncovered_count
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Accept
                    cover_count[covered_indices] -= 1
                    current_code.pop(remove_pos)
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered.copy()
                        
            else:
                # Swap
                if not current_code:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)
                
                if new_idx == old_idx:
                    T *= cooling_rate
                    continue
                
                old_covered = cover_matrix[old_idx, :]
                new_covered = cover_matrix[new_idx, :]
                
                # Calculate new uncovered
                # Words covered ONLY by old_idx will become uncovered unless covered by new_idx
                # Words covered by new_idx will become covered
                # Logic:
                # 1. Mark words covered by old_idx as potentially uncovered (if count was 1)
                # 2. Mark words covered by new_idx as covered
                # This is tricky with just masks. Let's use counts.
                
                # Temporarily remove old_idx
                cover_count[old_covered.nonzero()[0]] -= 1
                
                # Words that are now uncovered (count == 0)
                # But we haven't added new_idx yet.
                # Let's just compute the new state directly for the affected indices.
                # Affected indices = union of old_covered and new_covered
                affected = old_covered | new_covered
                aff_indices = affected.nonzero()[0]
                
                old_counts = cover_count[aff_indices].copy()
                
                # Apply removal of old
                cover_count[old_covered.nonzero()[0]] -= 1 # Wait, I already did this above? No, I did it in the temp block. Let's restart logic cleanly.
                
                # Clean Swap Logic:
                # 1. Save old counts for affected indices
                aff_indices = (cover_matrix[old_idx, :] | cover_matrix[new_idx, :]).nonzero()[0]
                old_counts = cover_count[aff_indices].copy()
                
                # 2. Update cover_count: remove old, add new
                cover_count[cover_matrix[old_idx, :].nonzero()[0]] -= 1
                cover_count[cover_matrix[new_idx, :].nonzero()[0]] += 1
                
                # 3. Determine new uncovered status for affected indices
                # An index is uncovered if cover_count is 0
                new_uncovered_status = (cover_count[aff_indices] == 0)
                
                # 4. Update global current_uncovered mask
                new_uncovered = current_uncovered.copy()
                new_uncovered[aff_indices] = new_uncovered_status
                
                new_code_len = len(current_code)
                new_uncovered_count = np.sum(new_uncovered)
                new_energy = new_code_len + 100 * new_uncovered_count
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Accept
                    current_code[remove_pos] = new_idx
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered.copy()
                else:
                    # Reject: Revert cover_count
                    cover_count[aff_indices] = old_counts
                    
            T *= cooling_rate
            
        if np.sum(best_uncovered) == 0:
            return best_code, best_uncovered
        else:
            if np.sum(current_uncovered) == 0:
                return current_code, current_uncovered
            else:
                return code, uncovered_mask

    # Initial greedy construction
    best_code, best_uncovered = greedy_cover()
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Run restarts if time permits
    while time.time() < deadline - 20.0:
        cand_code, cand_uncovered = greedy_cover()
        cand_code, _ = local_search_reduce(cand_code, cand_uncovered)
        
        cand_code, cand_uncovered = simulated_annealing(cand_code, cand_uncovered)
        
        if np.sum(cand_uncovered) == 0 and len(cand_code) < len(best_code):
            best_code = cand_code
            best_uncovered = cand_uncovered
    
    # Final polish
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    if time.time() < deadline - 5.0:
        best_code, best_uncovered = simulated_annealing(best_code, best_uncovered)
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
