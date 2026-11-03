import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
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
    
    # Precompute cover matrix
    # cover_matrix[i, j] is 1 if dist(i, j) <= radius
    # Using NumPy for speed
    cover_matrix = np.zeros((total_space, total_space), dtype=np.int8)
    
    # Efficiently build cover matrix
    # For radius 1, neighbors are self + change one position
    # We can vectorize this
    
    # Create an array of all words as a 2D array (N, n)
    words_arr = np.array(all_words)
    
    # Self coverage
    cover_matrix[np.arange(total_space), np.arange(total_space)] = 1
    
    if radius >= 1:
        for pos in range(n):
            # For each position, we want to add coverage between i and j
            # if they differ only at pos
            # This is a bit tricky to vectorize directly without loops over positions
            # But n=7 is small.
            
            # Alternative: For each word i, generate its neighbors and set bits.
            # This is O(N * n * q).
            # N=2187, n=7, q=3. 2187 * 7 * 2 = ~30k ops. Very fast.
            pass
            
    # Let's just use the loop method to fill cover_matrix, it's fast enough
    # and simpler to get right.
    cover_matrix = np.zeros((total_space, total_space), dtype=np.int8)
    for i in range(total_space):
        w = all_words[i]
        cover_matrix[i, i] = 1
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    cover_matrix[i, j] = 1

    # Helper for local search
    def local_search(code):
        if not code:
            return code
            
        # cover_count[j] = number of codewords in code covering j
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code:
            cover_count[cover_matrix[idx, :]] += 1
            
        # Reduction phase
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                return code
            improved = False
            # Shuffle to break symmetry
            random.shuffle(code)
            for i in range(len(code)):
                idx = code[i]
                # Check if idx is redundant
                # i.e., all words covered by idx are covered by at least 2 codewords
                # cover_count[cover_matrix[idx, :]] > 1
                if np.all(cover_count[cover_matrix[idx, :]] > 1):
                    # Remove idx
                    cover_count[cover_matrix[idx, :]] -= 1
                    code.pop(i)
                    improved = True
                    break # Restart loop to check other removals
                    
        # Swap phase
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                return code
            improved = False
            random.shuffle(code)
            for i in range(len(code)):
                idx = code[i]
                # Find words uniquely covered by idx
                unique_mask = (cover_count == 1) & cover_matrix[idx, :]
                if not np.any(unique_mask):
                    continue # Redundant, but reduction phase should have caught it
                    # If not caught, maybe due to order. Try to remove?
                    # But if unique_mask is empty, it is redundant.
                    # Let's try to remove it.
                    cover_count[cover_matrix[idx, :]] -= 1
                    code.pop(i)
                    improved = True
                    break
                
                critical_words = np.where(unique_mask)[0]
                # Find candidate k not in code that covers all critical_words
                # cover_matrix[k, critical_words].all()
                
                # To speed up, we can precompute for each word the set of critical words it covers?
                # No, just iterate.
                # We can vectorize the check for all k?
                # candidate_gains = cover_matrix[:, critical_words].all(axis=1)
                # This is (N, len(critical_words)).
                # len(critical_words) is small (usually).
                
                # Exclude current code
                in_code = np.zeros(total_space, dtype=bool)
                for c in code:
                    in_code[c] = True
                
                valid_candidates = np.where(candidate_gains := cover_matrix[:, critical_words].all(axis=1))[0]
                valid_candidates = valid_candidates[~in_code[valid_candidates]]
                
                if len(valid_candidates) > 0:
                    # Pick random candidate
                    new_idx = valid_candidates[random.randint(0, len(valid_candidates)-1)]
                    # Swap
                    cover_count[cover_matrix[idx, :]] -= 1
                    cover_count[cover_matrix[new_idx, :]] += 1
                    code[i] = new_idx
                    improved = True
                    break
                    
        return code

    def greedy_cover():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break
            # Calculate gains
            # gains[i] = number of uncovered words covered by i
            gains = cover_matrix[:, uncovered].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Randomize ties
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(idx)
            uncovered[cover_matrix[idx, :]] = False
            
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 20.0:
        code = greedy_cover()
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Final polish
    if best_code:
        best_code = local_search(best_code)
        
    return [all_words[i] for i in best_code]
