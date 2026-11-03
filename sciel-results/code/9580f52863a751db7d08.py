import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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
    
    # Precompute neighbors and reverse cover lists
    # neighbors[i] is a list of indices of words within distance <= radius of word i
    # reverse_cover[j] is a list of indices of codewords that cover word j
    neighbors = [[] for _ in range(total_space)]
    reverse_cover = [[] for _ in range(total_space)]
    
    # For radius 1, neighbors are self + all single-bit changes
    for i in range(total_space):
        w = all_words[i]
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    neighbors[i].append(j)
                    reverse_cover[j].append(i)

    # Build numpy cover matrix for fast vectorized operations
    # cover_matrix[i, j] is True if word j is covered by codeword i
    cover_matrix = np.zeros((total_space, total_space), dtype=np.uint8)
    for i in range(total_space):
        for j in neighbors[i]:
            cover_matrix[i, j] = 1
    
    # Also keep the list of neighbors for local search which might be faster for sparse updates
    # But numpy matrix is good for bulk operations like checking removability

    def greedy_cover(randomized=False, alpha=1.0):
        """
        Greedy set cover with optional randomization.
        Uses numpy for gain calculation.
        """
        uncovered = np.ones(total_space, dtype=np.uint8)
        code = []
        cm_view = cover_matrix # uint8
        
        while uncovered.any():
            # Calculate gain for each codeword: number of uncovered words it covers
            # gain[i] = sum(cover_matrix[i] * uncovered)
            # Since cover_matrix is 0/1 and uncovered is 0/1, this is a dot product
            gains = cm_view.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if space is connected
                
            # Find candidates with max gain
            # For randomization, we can pick from top candidates or use a threshold
            if randomized:
                # Pick from candidates with gain >= 0.8 * max_gain
                threshold = max_gain * 0.8
                candidates = np.where(gains >= threshold)[0]
            else:
                candidates = np.where(gains == max_gain)[0]
                
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            code.append(int(best_idx))
            
            # Update uncovered
            uncovered[cover_matrix[best_idx] == 1] = 0
            
        return code

    def local_search_reduce(code):
        """
        Local search to minimize code size.
        1. Remove redundant codewords.
        2. Swap codewords if it leads to better structure (potential for future removals).
        """
        if not code:
            return code
            
        current_code = list(code)
        
        # Build cover_count: cover_count[w] = number of codewords covering w
        # Initialize to 0
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Update cover_count
        for idx in current_code:
            cover_count += cover_matrix[idx]
            
        improved = True
        max_iterations = 2000
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # 1. Try to remove redundant codewords
            # A codeword c is redundant if all words it covers are covered by at least one other codeword.
            # i.e., for all w in neighbors[c], cover_count[w] > 1.
            
            # We iterate over a copy of the list to allow deletion
            # Shuffle to break symmetry
            code_list = current_code[:]
            random.shuffle(code_list)
            
            removed_any = False
            for idx in code_list:
                # Check if removable
                # Get cover counts for words covered by idx
                # mask = cover_matrix[idx] == 1
                # counts = cover_count[mask]
                # if counts.min() > 1: removable
                
                # Vectorized check
                mask = cover_matrix[idx] == 1
                if np.all(cover_count[mask] > 1):
                    # Removable
                    # Decrement cover_count
                    cover_count[mask] -= 1
                    current_code.remove(idx)
                    improved = True
                    removed_any = True
                    break # Restart loop with updated counts
            
            if removed_any:
                continue
                
            # 2. Try to swap codewords
            # If we cannot remove any, try to find a swap that reduces the number of "private" words
            # or simply finds a valid swap if it leads to a smaller set (which is impossible in a swap 1-for-1)
            # Actually, a swap 1-for-1 keeps the size same. 
            # To reduce size, we need to remove 1 and add 0 (impossible if not redundant) 
            # OR remove 1 and add 1, but then later remove another? 
            # Standard approach: 
            # Find a codeword c with private words. 
            # Find a candidate c' that covers all private words of c.
            # If we swap c with c', size remains same. 
            # However, this might make c' redundant later, or make other codewords redundant.
            
            # Better strategy for size reduction:
            # If no removal is possible, we are in a local minimum for size.
            # We need to perturb.
            
            # For now, just try to find a swap that might help future removals.
            # Identify codewords with private words.
            swaps_found = False
            for idx in code_list:
                # Find private words of idx
                mask = cover_matrix[idx] == 1
                private_words = np.where((cover_count[mask] == 1))[0]
                
                if len(private_words) == 0:
                    # This shouldn't happen if we just checked for removal, 
                    # but due to shuffle/order, maybe? No, if cover_count[w] > 1 for all w, it was removable.
                    # So if we are here, there is at least one private word.
                    pass
                
                if len(private_words) > 0:
                    # Find candidates that cover ALL private words
                    # Start with reverse_cover of first private word
                    # private_words is an array of indices into the 'mask' array? No, np.where returns indices into the original array if we don't specify.
                    # Wait, np.where((cover_count[mask] == 1))[0] gives indices into the flattened array of length total_space?
                    # No, cover_count[mask] creates a 1D array of length sum(mask).
                    # So private_words contains indices into that 1D array, NOT the global word indices.
                    # We need the global word indices.
                    
                    # Let's get the global indices of private words
                    global_private_indices = np.where(mask)[0][private_words]
                    
                    # Now find candidates that cover all these global_private_indices
                    # Intersection of reverse_cover for each
                    candidates = set(reverse_cover[global_private_indices[0]])
                    for w in global_private_indices[1:]:
                        candidates &= set(reverse_cover[w])
                        
                    # Exclude current code
                    candidates -= set(current_code)
                    
                    if candidates:
                        # Pick a random candidate
                        new_idx = random.choice(list(candidates))
                        
                        # Perform swap
                        # Remove idx
                        cover_count[cover_matrix[idx] == 1] -= 1
                        current_code.remove(idx)
                        
                        # Add new_idx
                        cover_count[cover_matrix[new_idx] == 1] += 1
                        current_code.append(new_idx)
                        
                        improved = True
                        swaps_found = True
                        break # Restart loop with updated counts
            
            if not improved:
                # No removal, no swap found? 
                # This means for every codeword, there is no single codeword outside the set 
                # that covers all its private words.
                # We are stuck. Return current best.
                break
                
        return current_code

    def iterated_local_search(initial_code):
        current_code = local_search_reduce(initial_code)
        best_code = current_code[:]
        best_size = len(best_code)
        
        # Simulated Annealing / Iterated Local Search
        temperature = 10.0
        min_temperature = 1.0
        cooling = 0.95
        
        while time.time() < deadline - 5.0:
            # Perturbation
            # Remove k random codewords
            k = random.randint(1, 3)
            if len(current_code) <= k:
                k = max(1, len(current_code) // 2)
            if k == 0:
                break
                
            to_remove = random.sample(current_code, k)
            to_remove_set = set(to_remove)
            perturbed = [c for c in current_code if c not in to_remove_set]
            
            # Repair
            # Compute uncovered
            covered_mask = np.zeros(total_space, dtype=np.uint8)
            for c in perturbed:
                covered_mask |= cover_matrix[c]
            uncovered = (covered_mask == 0)
            
            if uncovered.any():
                # Greedy repair
                code = perturbed[:]
                unc_np = uncovered.astype(np.uint8)
                while unc_np.any():
                    gains = cover_matrix.dot(unc_np)
                    # Exclude already chosen? 
                    # We should only add new codewords.
                    # Mask out chosen ones
                    for c in code:
                        gains[c] = 0
                        
                    best_gain = np.max(gains)
                    if best_gain == 0:
                        break
                    candidates = np.where(gains == best_gain)[0]
                    chosen = candidates[random.randint(0, len(candidates)-1)]
                    code.append(int(chosen))
                    unc_np[cover_matrix[chosen] == 1] = 0
            else:
                code = perturbed[:]
                
            # Optimize
            repaired = local_search_reduce(code)
            
            # Acceptance
            new_size = len(repaired)
            if new_size < best_size:
                best_code = repaired
                best_size = new_size
                current_code = best_code[:]
                # Reset temperature on improvement?
                temperature = min(temperature + 2.0, 10.0)
            else:
                # Accept worse with probability based on temperature
                delta = new_size - best_size
                if delta == 0:
                    # Same size, accept to explore
                    if random.random() < 0.5:
                        current_code = repaired
                else:
                    # Worse
                    # prob = exp(-delta / temperature)
                    prob = 2.71828 ** (-delta / temperature)
                    if random.random() < prob:
                        current_code = repaired
                    else:
                        # Revert to best? Or keep perturbed? 
                        # Usually keep best to avoid drifting too far.
                        current_code = best_code[:]
                        
            temperature *= cooling
            if temperature < min_temperature:
                temperature = min_temperature
                
        return best_code

    # Main Loop
    best_code = None
    best_size = float('inf')
    
    # Initial deterministic greedy
    init_code = greedy_cover(randomized=False)
    opt_code = local_search_reduce(init_code)
    best_code = opt_code
    best_size = len(best_code)
    
    # Iterated Local Search on the best found so far
    if time.time() < deadline - 10.0:
        ils_code = iterated_local_search(best_code)
        if len(ils_code) < best_size:
            best_code = ils_code
            best_size = len(best_code)
            
    # Randomized Restarting
    # Try different random seeds for greedy
    while time.time() < deadline - 10.0:
        # Randomized greedy
        rand_code = greedy_cover(randomized=True)
        rand_code = local_search_reduce(rand_code)
        
        if len(rand_code) < best_size:
            best_code = rand_code
            best_size = len(best_code)
            # If we improved, do a quick ILS on this new best
            if time.time() < deadline - 15.0:
                improved_ils = iterated_local_search(best_code)
                if len(improved_ils) < best_size:
                    best_code = improved_ils
                    best_size = len(best_code)
        else:
            # If not improved, maybe do a short ILS on this candidate to see if it leads to a better basin?
            # Only if we have time
            if time.time() < deadline - 20.0 and random.random() < 0.1:
                short_ils = iterated_local_search(rand_code)
                if len(short_ils) < best_size:
                    best_code = short_ils
                    best_size = len(best_code)

    # Final polish: try to remove any remaining redundancies in the best code
    final_code = local_search_reduce(best_code)
    
    return [all_words[i] for i in final_code]

if __name__ == "__main__":
    # Self-test for q=3, n=7, radius=1
    # construct(3, 7, 1)
    pass
