import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    total_space = q ** n
    neighbor_count = 1 + n * (q - 1)  # For radius=1: 1 + 7*2 = 15

    # Precompute neighbors for each word using vectorized approach
    # neighbors[i] is a list of indices of words within Hamming distance <= radius of i
    neighbors = np.zeros((total_space, neighbor_count), dtype=np.int32)
    
    # Precompute all words as index -> list of digits
    # Use a more efficient approach: precompute the base word digits
    all_words = np.zeros((total_space, n), dtype=np.int8)
    for i in range(total_space):
        tmp = i
        for pos in range(n):
            all_words[i, pos] = tmp % q
            tmp //= q
    
    # Build neighbors array
    # For each word, its neighbors are itself + all words differing in exactly one position
    # For position p and new value v (v != original), compute the index
    neighbors[:, 0] = np.arange(total_space)
    
    col = 1
    for pos in range(n):
        orig_vals = all_words[:, pos]
        for v in range(q):
            # Skip if v equals original value
            mask = orig_vals != v
            if not np.any(mask):
                continue
            # For words where original value != v, compute new index
            # New index = old_index + (v - orig_val) * q^(n-1-pos)
            multiplier = q ** (n - 1 - pos)
            new_indices = np.arange(total_space, dtype=np.int32) + (v - orig_vals) * multiplier
            neighbors[mask, col] = new_indices[mask]
            col += 1
    
    # Verify col should equal neighbor_count
    # For q=3, n=7, radius=1: col should be 15
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    def greedy_cover(randomize=False, seed=None):
        """Greedy set cover to find initial covering code."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = random
        
        covered = np.zeros(total_space, dtype=bool)
        code = []
        
        while not np.all(covered):
            gains = np.sum(~covered[neighbors], axis=1)
            
            if randomize:
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 0:
                    best_idx = candidates[local_rng.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
            
            code.append(int(best_idx))
            covered[neighbors[best_idx]] = True
            
        return code

    def reduce_code(code, time_limit=None):
        """Remove redundant codewords from code. Returns reduced list."""
        if not code:
            return code
        
        if time_limit is None:
            time_limit = deadline
        
        current_code = list(code)
        
        if len(current_code) == 0:
            return current_code
            
        flat_neighbors = neighbors[current_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > time_limit:
                    return current_code
                
                idx = current_code[i]
                min_count = np.min(cover_count[neighbors[idx]])
                
                if min_count > 1:
                    cover_count[neighbors[idx]] -= 1
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    def swap_local_search(code, max_iterations=200, time_limit=None):
        """
        Perform swap-based local search to improve the covering code.
        Try removing a codeword and replacing it with a better one, then reduce.
        """
        if not code:
            return code
        
        if time_limit is None:
            time_limit = deadline
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        if best_len == 0:
            return best_code
            
        # Build cover_count for current_code
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        for iteration in range(max_iterations):
            if time.time() > time_limit:
                break
            
            # Pick a random codeword to swap out
            i = random.randint(0, len(current_code) - 1)
            idx_out = current_code[i]
            
            # Tentatively remove idx_out
            cover_count[neighbors[idx_out]] -= 1
            current_code.pop(i)
            
            # Find uncovered words
            uncovered_mask = (cover_count == 0)
            uncovered_count = np.sum(uncovered_mask)
            
            if uncovered_count == 0:
                # It was redundant - restore and continue
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Find best replacement: maximize coverage of uncovered words
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude words already in current_code
            for cw in current_code:
                gains[cw] = -1
            
            best_j = np.argmax(gains)
            
            if gains[best_j] <= 0:
                # No good replacement found, revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                continue
            
            # Add best_j
            current_code.append(int(best_j))
            cover_count[neighbors[best_j]] += 1
            
            # Now try to reduce the new set
            new_code = list(current_code)
            # Rebuild cover_count for new_code to be safe
            flat_new = neighbors[new_code].flatten()
            new_cover_count = np.bincount(flat_new, minlength=total_space)
            
            # Try to remove redundant words
            improved = True
            while improved:
                improved = False
                random.shuffle(new_code)
                for k in range(len(new_code)):
                    idx = new_code[k]
                    if np.min(new_cover_count[neighbors[idx]]) > 1:
                        new_cover_count[neighbors[idx]] -= 1
                        new_code.pop(k)
                        improved = True
                        break
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                # Update current state to best
                current_code = new_code
                flat_best = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_best, minlength=total_space)
            else:
                # Revert
                current_code.insert(i, idx_out)
                cover_count[neighbors[idx_out]] += 1
                # Remove best_j from current_code
                if best_j in current_code:
                    current_code.remove(best_j)
                cover_count[neighbors[best_j]] -= 1
        
        return best_code

    def two_opt_local_search(code, max_iterations=100, time_limit=None):
        """
        Try 2-opt style moves: remove two codewords and add two better ones.
        This can escape local minima that 1-swap cannot.
        """
        if not code:
            return code
        
        if time_limit is None:
            time_limit = deadline
        
        current_code = list(code)
        best_code = current_code[:]
        best_len = len(current_code)
        
        if best_len < 2:
            return best_code
            
        flat_neighbors = neighbors[best_code].flatten()
        cover_count = np.bincount(flat_neighbors, minlength=total_space)
        
        for iteration in range(max_iterations):
            if time.time() > time_limit:
                break
            
            # Pick two random codewords to swap out
            i, j = random.sample(range(len(current_code)), 2)
            idx_out1 = current_code[i]
            idx_out2 = current_code[j]
            
            # Tentatively remove both
            cover_count[neighbors[idx_out1]] -= 1
            cover_count[neighbors[idx_out2]] -= 1
            # Remove from list (be careful with indices)
            if i < j:
                current_code.pop(j)
                current_code.pop(i)
            else:
                current_code.pop(i)
                current_code.pop(j)
            
            # Find uncovered words
            uncovered_mask = (cover_count == 0)
            uncovered_count = np.sum(uncovered_mask)
            
            if uncovered_count == 0:
                # Both were redundant - restore and continue
                current_code.insert(i, idx_out1)
                current_code.insert(j, idx_out2)
                cover_count[neighbors[idx_out1]] += 1
                cover_count[neighbors[idx_out2]] += 1
                continue
            
            # Find best two replacements: maximize coverage of uncovered words
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            
            # Exclude words already in current_code
            for cw in current_code:
                gains[cw] = -1
            
            # Find top 2 candidates
            sorted_indices = np.argsort(gains)[::-1]
            candidates = []
            for idx in sorted_indices[:10]:
                if gains[idx] > 0:
                    candidates.append(int(idx))
                if len(candidates) >= 2:
                    break
            
            if len(candidates) < 2:
                # Not enough good candidates, revert
                current_code.insert(i, idx_out1)
                current_code.insert(j, idx_out2)
                cover_count[neighbors[idx_out1]] += 1
                cover_count[neighbors[idx_out2]] += 1
                continue
            
            # Add the two best candidates
            for cand in candidates:
                current_code.append(cand)
                cover_count[neighbors[cand]] += 1
            
            # Try to reduce
            new_code = list(current_code)
            flat_new = neighbors[new_code].flatten()
            new_cover_count = np.bincount(flat_new, minlength=total_space)
            
            improved = True
            while improved:
                improved = False
                random.shuffle(new_code)
                for k in range(len(new_code)):
                    idx = new_code[k]
                    if np.min(new_cover_count[neighbors[idx]]) > 1:
                        new_cover_count[neighbors[idx]] -= 1
                        new_code.pop(k)
                        improved = True
                        break
            
            if len(new_code) < best_len:
                best_len = len(new_code)
                best_code = new_code
                current_code = new_code
                flat_best = neighbors[best_code].flatten()
                cover_count = np.bincount(flat_best, minlength=total_space)
            else:
                # Revert
                current_code.insert(i, idx_out1)
                current_code.insert(j, idx_out2)
                cover_count[neighbors[idx_out1]] += 1
                cover_count[neighbors[idx_out2]] += 1
                for cand in candidates:
                    current_code.remove(cand)
                    cover_count[neighbors[cand]] -= 1
        
        return best_code

    # Strategy:
    # 1. Run deterministic greedy + reduce.
    # 2. Run randomized greedy + reduce + swap local search multiple times.
    # 3. Try 2-opt moves to escape local minima.
    # 4. Keep the best.

    best_code = greedy_cover(randomize=False)
    best_code = reduce_code(best_code)
    best_len = len(best_code)
    
    # Try swap local search on the initial solution
    improved_code = swap_local_search(best_code, max_iterations=150)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code
    
    # Try 2-opt on the best so far
    improved_code = two_opt_local_search(best_code, max_iterations=50)
    if len(improved_code) < best_len:
        best_len = len(improved_code)
        best_code = improved_code
    
    # Randomized restarts with more aggressive local search
    num_restarts = 0
    while time.time() < deadline - 10.0:
        seed = num_restarts * 7919 + 1
        cand_code = greedy_cover(randomize=True, seed=seed)
        cand_code = reduce_code(cand_code)
        
        # Apply swap local search
        cand_code = swap_local_search(cand_code, max_iterations=80)
        
        # Apply 2-opt if there's time
        if time.time() < deadline - 5.0:
            cand_code = two_opt_local_search(cand_code, max_iterations=30)
        
        num_restarts += 1
        
        if len(cand_code) < best_len:
            best_len = len(cand_code)
            best_code = cand_code
    
    # Final polish: try more aggressive reduction and swaps on the best code
    if time.time() < deadline - 3.0:
        final_code = reduce_code(best_code)
        if len(final_code) < best_len:
            best_len = len(final_code)
            best_code = final_code
        
        if time.time() < deadline - 1.0:
            final_code = swap_local_search(best_code, max_iterations=50)
            if len(final_code) < best_len:
                best_len = len(final_code)
                best_code = final_code
    
    # Convert indices back to words
    return [idx_to_word(i) for i in best_code]
