import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Convert word index to tuple of digits
    def index_to_word(idx):
        return [idx // (q ** (n - i - 1)) % q for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for all words
    neighbors = [[] for _ in range(total_words)]
    neighbor_sets = [set() for _ in range(total_words)]
    
    for i in range(total_words):
        w = index_to_word(i)
        neighbors[i].append(i)
        neighbor_sets[i].add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    j = word_to_index(w)
                    neighbors[i].append(j)
                    neighbor_sets[i].add(j)
            w[pos] = orig_val
            
    # Convert to arrays for fast numpy operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Initial code: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.append(word_to_index(word))
            
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            counts[neighbor_arrays[c]] += 1
        return counts
        
    def greedy_reduce(code_indices, rng):
        current_code = list(code_indices)
        coverage_count = compute_counts(current_code)
        
        while True:
            redundant_indices = []
            
            for idx, c in enumerate(current_code):
                is_redundant = True
                for w in neighbors[c]:
                    if coverage_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_indices.append(idx)
                    
            if not redundant_indices:
                break
                
            rem_idx = redundant_indices[rng.randint(0, len(redundant_indices) - 1)]
            c_to_remove = current_code[rem_idx]
            current_code.pop(rem_idx)
            
            for w in neighbors[c_to_remove]:
                coverage_count[w] -= 1
                
        return current_code

    def local_search_with_swaps(code_indices, rng, time_limit):
        current_code = list(code_indices)
        code_set = set(current_code)
        counts = compute_counts(current_code)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # --- Deletion Pass ---
            to_remove = []
            for c in current_code:
                if np.min(counts[neighbor_arrays[c]]) > 1:
                    to_remove.append(c)
                    
            if to_remove:
                improved = True
                for c in to_remove:
                    current_code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
            
            # --- Swap Pass ---
            random.shuffle(current_code)
            
            for c in current_code:
                if time.time() > time_limit:
                    break
                    
                if np.min(counts[neighbor_arrays[c]]) > 1:
                    continue
                    
                crit_points = [i for i in neighbors[c] if counts[i] == 1]
                
                if not crit_points:
                    continue
                
                found_swap = False
                random.shuffle(crit_points)
                
                for p in crit_points:
                    if found_swap:
                        break
                        
                    cands = [x for x in neighbors[p] if x not in code_set]
                    
                    best_w = -1
                    best_score = -1
                    
                    for w in cands:
                        valid = True
                        for k in crit_points:
                            if k not in neighbor_sets[w]:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        score = np.sum(counts[neighbor_arrays[w]] == 2)
                        
                        if score > best_score:
                            best_score = score
                            best_w = w
                            
                    if best_w != -1:
                        current_code.remove(c)
                        code_set.remove(c)
                        counts[neighbor_arrays[c]] -= 1
                        
                        current_code.append(best_w)
                        code_set.add(best_w)
                        counts[neighbor_arrays[best_w]] += 1
                        
                        improved = True
                        found_swap = True
                        break
                        
        return current_code

    def greedy_set_cover(rng, time_limit):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered) and time.time() < time_limit:
            gains = np.zeros(total_words, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                gains[neighbors[idx]] += 1
                
            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]
            
            if max_gain == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
            
        return code

    # Main search loop
    best_code = initial_code
    best_len = len(initial_code)
    
    rng = random.Random(42)
    
    # First, try pure greedy on the initial code
    reduced = greedy_reduce(list(initial_code), rng)
    if len(reduced) < best_len:
        best_len = len(reduced)
        best_code = reduced
    
    # Then, try local search with swaps on the best code found so far
    if time.time() < deadline - 10:
        swapped = local_search_with_swaps(best_code, rng, deadline - 5)
        if len(swapped) < best_len:
            best_len = len(swapped)
            best_code = swapped
    
    # Continue with more iterations if time permits
    while time.time() < deadline - 10:
        # Generate a perturbed version of the best code
        perturbed = list(best_code)
        
        # Randomly remove a few codewords and re-add random ones
        num_remove = rng.randint(1, min(3, len(perturbed)))
        for _ in range(num_remove):
            if perturbed:
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Add random codewords to ensure coverage
        for _ in range(num_remove + rng.randint(0, 2)):
            perturbed.append(rng.randint(0, total_words - 1))
        
        # Reduce
        reduced = greedy_reduce(perturbed, rng)
        
        # Local search
        if time.time() < deadline - 5:
            swapped = local_search_with_swaps(reduced, rng, deadline - 5)
            if len(swapped) < best_len:
                best_len = len(swapped)
                best_code = swapped
        else:
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_code = reduced
    
    # Convert best_code indices back to lists of integers
    final_codewords = []
    for idx in best_code:
        final_codewords.append(index_to_word(idx))
        
    return final_codewords
