import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    num_coords = n
    alphabet = q
    cover_size = 1 + num_coords * (alphabet - 1)

    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    # Vectorized neighbor precomputation
    # neighbors[w, j] = the word at Hamming distance <= 1 from w (including w itself)
    neighbors = np.zeros((total_words, cover_size), dtype=np.int32)
    neighbors[:, 0] = np.arange(total_words)
    
    # For each position, compute the neighbors by changing that digit
    for pos in range(num_coords):
        # Power of alphabet for this position
        pow_a = alphabet ** (num_coords - 1 - pos)
        # For each word, get its digit at this position
        # We can compute this using modular arithmetic
        # word // pow_a % alphabet gives the digit at position pos
        
        # Generate all words as a column vector
        words_col = np.arange(total_words, dtype=np.int32)
        digits_at_pos = (words_col // pow_a) % alphabet
        
        # For each word and each alternative value, compute the neighbor
        # neighbor = word - digit * pow_a + new_val * pow_a
        # = word + (new_val - digit) * pow_a
        
        # We need to fill columns 1, 2, 3, ... for each position
        col_start = 1 + pos * (alphabet - 1)
        for new_val in range(alphabet):
            # Skip if new_val equals the current digit
            # We'll compute all and then mask
            delta = (new_val - digits_at_pos) * pow_a
            # Only valid where delta != 0
            valid_mask = (delta != 0)
            neighbor_vals = words_col + delta
            # Clip to valid range (should be fine since we're changing one digit)
            # Place into the appropriate column
            # We need to figure out which column index for each word
            # The column for word w changing position pos to new_val is:
            # col_start + (number of values < new_val that are != digits_at_pos[w])
            # This is tricky to vectorize simply. Let's do it differently.
            
            # Alternative: for each word, for each pos, for each val != digit, compute neighbor
            # Let's just loop over words but vectorize the inner part
            pass
    
    # Simpler approach: loop over positions and values, vectorize over words
    for pos in range(num_coords):
        pow_a = alphabet ** (num_coords - 1 - pos)
        words_col = np.arange(total_words, dtype=np.int32)
        digits_at_pos = (words_col // pow_a) % alphabet
        
        col_idx = 1
        for new_val in range(alphabet):
            # For words where digits_at_pos != new_val, compute neighbor
            mask = (digits_at_pos != new_val)
            if np.any(mask):
                delta = (new_val - digits_at_pos[mask]) * pow_a
                neighbor_vals = words_col[mask] + delta
                # Place into neighbors[words_col[mask], col_idx]
                neighbors[words_col[mask], col_idx] = neighbor_vals
            col_idx += 1

    # Verify: neighbors[w] should contain w and all its distance-1 neighbors
    # covers_word is the same as neighbors (symmetric relation)
    covers_word = neighbors  # alias, same data

    # Helper: Local Search with removals and swaps
    def local_search(cw_list, cover_count):
        """cw_list: list of codeword indices, cover_count: np.array of coverage counts"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_cw = len(cw_list)
            
            # Try removals first
            for i in range(n_cw):
                cw = cw_list[i]
                removable = True
                for j in range(cover_size):
                    w_idx = neighbors[cw, j]
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove cw
                    cw_list.pop(i)
                    for j in range(cover_size):
                        w_idx = neighbors[cw, j]
                        cover_count[w_idx] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add another
            for i in range(n_cw):
                cw_out = cw_list[i]
                # Check if removing cw_out leaves any word with count 0
                # Those words need to be covered by the replacement
                holes = []
                for j in range(cover_size):
                    w_idx = neighbors[cw_out, j]
                    if cover_count[w_idx] == 1:
                        holes.append(w_idx)
                
                if not holes:
                    # Can just remove
                    cw_list.pop(i)
                    for j in range(cover_size):
                        w_idx = neighbors[cw_out, j]
                        cover_count[w_idx] -= 1
                    improved = True
                    break
                
                # Find a replacement that covers all holes
                # Candidates: neighbors of the hole words
                # Collect all candidates from holes
                cand_set = set()
                for h in holes:
                    for j in range(cover_size):
                        cand_set.add(int(neighbors[h, j]))
                
                # Remove cw_out from candidates (it's already in the set)
                cand_set.discard(cw_out)
                
                best_cand = -1
                best_gain = -1
                for cand in cand_set:
                    # Check if cand covers all holes
                    covers_all = True
                    gain = 0
                    for j in range(cover_size):
                        w_idx = neighbors[cand, j]
                        if w_idx in holes:
                            # This hole is covered
                            pass
                        if cover_count[w_idx] == 0:
                            # Adding cand would cover an uncovered word (not a hole, but a word with 0)
                            # Actually, after removing cw_out, some words have count 0 (the holes)
                            # We need cand to cover all holes
                            pass
                    
                    # More efficient: check coverage of holes
                    hole_set = set(holes)
                    for j in range(cover_size):
                        w_idx = int(neighbors[cand, j])
                        if w_idx in hole_set:
                            hole_set.discard(w_idx)
                    
                    if not hole_set:
                        # cand covers all holes
                        # Compute gain: number of words with count 0 that cand covers
                        # (excluding the holes which are already accounted for)
                        gain = 0
                        for j in range(cover_size):
                            w_idx = int(neighbors[cand, j])
                            if cover_count[w_idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                
                if best_cand != -1:
                    # Perform swap
                    cw_list[i] = best_cand
                    # Update cover_count: remove cw_out, add best_cand
                    for j in range(cover_size):
                        w_idx = neighbors[cw_out, j]
                        cover_count[w_idx] -= 1
                    for j in range(cover_size):
                        w_idx = neighbors[best_cand, j]
                        cover_count[w_idx] += 1
                    improved = True
                    break
        
        return cw_list, cover_count

    # Helper: Greedy Initialization
    def greedy_init():
        # Count how many uncovered words each candidate covers
        # Initially all words are uncovered, so count[cw] = cover_size for all cw
        # But we want to pick the one that covers the most *currently uncovered* words
        count = np.full(total_words, cover_size, dtype=np.int32)
        uncovered_mask = np.ones(total_words, dtype=np.uint8)
        
        cw_list = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        num_covered = 0
        
        while num_covered < total_words:
            if time.time() > deadline:
                break
            best_cw = int(np.argmax(count))
            if count[best_cw] == 0:
                break
            
            cw_list.append(best_cw)
            for j in range(cover_size):
                w_idx = int(neighbors[best_cw, j])
                if uncovered_mask[w_idx]:
                    uncovered_mask[w_idx] = 0
                    num_covered += 1
                    cover_count[w_idx] += 1
                    # Decrement count of all candidates that cover this word
                    for k in range(cover_size):
                        cw_idx = int(neighbors[w_idx, k])
                        count[cw_idx] -= 1
                else:
                    cover_count[w_idx] += 1
        
        return cw_list, cover_count

    # Helper: Perturb and Repair
    def perturb_and_repair(cw_list, cover_count, num_remove=5):
        new_list = list(cw_list)
        new_cover = cover_count.copy()
        
        # Remove random codewords
        to_remove = random.sample(range(len(new_list)), min(num_remove, len(new_list)))
        # Sort in reverse to avoid index issues
        to_remove.sort(reverse=True)
        removed_cws = []
        for idx in to_remove:
            cw = new_list[idx]
            new_list.pop(idx)
            removed_cws.append(cw)
            for j in range(cover_size):
                w_idx = int(neighbors[cw, j])
                new_cover[w_idx] -= 1
        
        # Repair: add codewords to cover uncovered words
        while True:
            if time.time() > deadline:
                return None, None
            
            # Find uncovered words
            uncovered_indices = np.where(new_cover == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            w = int(uncovered_indices[0])
            
            # Get candidates for w
            candidates = neighbors[w]
            
            best_cand = -1
            best_gain = -1
            
            for c in candidates:
                c = int(c)
                # Calculate gain: number of words in neighbors[c] that are currently uncovered
                gain = 0
                for j in range(cover_size):
                    w_idx = int(neighbors[c, j])
                    if new_cover[w_idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            
            if best_cand == -1:
                break
            
            new_list.append(best_cand)
            for j in range(cover_size):
                w_idx = int(neighbors[best_cand, j])
                new_cover[w_idx] += 1
        
        return new_list, new_cover

    # 1. Initial Greedy
    best_list, best_cover = greedy_init()
    best_list, best_cover = local_search(best_list, best_cover)
    best_size = len(best_list)
    
    # 2. Iterated Local Search
    stall_count = 0
    while time.time() < deadline - 5:
        if best_size <= 486:
            break
        
        # Adaptive perturbation
        if stall_count < 3:
            num_remove = random.randint(2, 4)
        elif stall_count < 6:
            num_remove = random.randint(4, 7)
        else:
            num_remove = random.randint(5, 10)
        
        perturbed_list, perturbed_cover = perturb_and_repair(best_list, best_cover, num_remove)
        
        if perturbed_list is None:
            break
        
        # Local Search on the perturbed solution
        perturbed_list, perturbed_cover = local_search(perturbed_list, perturbed_cover)
        
        if len(perturbed_list) < best_size:
            best_list = perturbed_list
            best_cover = perturbed_cover
            best_size = len(best_list)
            stall_count = 0
        else:
            stall_count += 1
    
    # Convert to list of digits
    result = [idx_to_digits(cw) for cw in best_list]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
