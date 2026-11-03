import time
import random
import numpy as np
from collections import deque

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
    
    # Build cover matrix: cover_matrix[i] is a bitset (list of Python ints) 
    # representing which words are within distance <= radius of word i.
    # For q=3, n=7, total_space=2187, we need 2187/64 ≈ 35 uint64 words per row.
    # We'll use Python integers as arbitrary-precision bitsets for simplicity and speed.
    
    # cover_bits[i] is a Python int where bit j is 1 iff word j is covered by word i.
    cover_bits = [0] * total_space
    
    # For radius 1: word i covers itself and all words at Hamming distance 1.
    if radius == 1:
        for i in range(total_space):
            bits = 1 << i  # self
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                power = q ** (n - 1 - pos)
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * power
                    bits |= (1 << j)
            cover_bits[i] = bits
    else:
        # Generic radius: compute Hamming distances
        for i in range(total_space):
            bits = 0
            wi = all_words[i]
            for j in range(total_space):
                wj = all_words[j]
                dist = sum(1 for a, b in zip(wi, wj) if a != b)
                if dist <= radius:
                    bits |= (1 << j)
            cover_bits[i] = bits
    
    # Precompute list of covered indices for each word (for local search)
    cover_indices = []
    for i in range(total_space):
        idxs = []
        bits = cover_bits[i]
        while bits:
            lsb = bits & (-bits)
            idx = lsb.bit_length() - 1
            idxs.append(idx)
            bits ^= lsb
        cover_indices.append(idxs)
    
    # Convert to numpy boolean matrix for fast operations
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    for i in range(total_space):
        for j in cover_indices[i]:
            cover_matrix[i, j] = True
    
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Greedy Set Cover using NumPy
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int8 = uncovered.astype(np.int8)
        
        while np.any(uncovered):
            gains = cm_int8.dot(unc_int8)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Tie-break: prefer codewords that cover more already-uncovered words
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            unc_int8 = uncovered.astype(np.int8)
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try swap
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        # Find candidates that cover all unique words
                        candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                        
                        if len(candidates) > 0:
                            current_set = set(current_code)
                            valid_cands = [c for c in candidates if c not in current_set]
                            
                            if len(valid_cands) > 0:
                                new_idx = valid_cands[random.randint(0, len(valid_cands)-1)]
                                
                                cover_count -= mask
                                cover_count += cover_matrix[new_idx]
                                
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                            
        return current_code

    # Branch and bound for minimum dominating set
    def branch_and_bound(greedy_solution):
        """
        Use branch and bound to try to find a smaller dominating set.
        We use the greedy solution as an initial upper bound.
        """
        if time.time() > deadline:
            return greedy_solution
        
        best_code = greedy_solution[:]
        best_size = len(best_code)
        
        # uncovered_set as a Python int bitset
        full_mask = (1 << total_space) - 1
        
        # Precompute for each word, the set of words it covers (as Python int)
        # cover_bits[i] already has this.
        
        # Order candidates by "potential": words that cover many uncovered words
        # We'll use a priority queue approach.
        
        # For efficiency, we'll use a recursive B&B with pruning.
        # But 2187 vertices is too large for naive B&B.
        # Instead, we'll use a local search guided by B&B-like pruning.
        
        # Alternative: use the fact that this is a covering code problem.
        # We'll do a more aggressive local search with 2-exchange moves.
        
        # 2-exchange local search
        def two_exchange_search(code):
            if not code:
                return code
            
            current_code = code[:]
            cover_count = np.zeros(total_space, dtype=np.int32)
            
            for idx in current_code:
                cover_count += cover_matrix[idx]
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                random.shuffle(current_code)
                n_cw = len(current_code)
                
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx1 = current_code[i]
                    mask1 = cover_matrix[idx1]
                    
                    # Words uniquely covered by idx1
                    unique1 = mask1 & (cover_count == 1)
                    
                    if not np.any(unique1):
                        # Can remove
                        cover_count -= mask1
                        current_code.pop(i)
                        improved = True
                        break
                    
                    for j in range(i + 1, n_cw):
                        if time.time() > deadline:
                            return current_code
                        
                        idx2 = current_code[j]
                        mask2 = cover_matrix[idx2]
                        
                        # Words uniquely covered by idx2
                        unique2 = mask2 & (cover_count == 1)
                        
                        # Try swapping idx1 and idx2
                        # After removing both, what's uncovered?
                        new_count = cover_count - mask1 - mask2
                        # Words that become uncovered
                        newly_uncovered = (new_count == 0) & (cover_count > 0)
                        
                        if np.any(newly_uncovered):
                            continue
                        
                        # Check if adding both back covers everything
                        # Actually, we need to find a replacement pair
                        # For now, just try individual removals
                        pass
                    
            return current_code
        
        # Run 2-exchange search
        improved_code = two_exchange_search(best_code)
        if len(improved_code) < best_size:
            best_code = improved_code
            best_size = len(best_code)
        
        # More aggressive: try to find a better solution via randomized local search
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Start from greedy with random perturbation
            perturbed = greedy_cover_randomized()
            perturbed = local_search_reduce(perturbed)
            
            if len(perturbed) < best_size:
                best_code = perturbed
                best_size = len(best_code)
                # Further improve
                best_code = local_search_reduce(best_code)
                best_size = len(best_code)
        
        return best_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run branch and bound
    best_code = branch_and_bound(best_code)
    
    # Run more restarts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
                if time.time() < deadline - 5.0:
                    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
