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

    # Precompute neighbors
    neighbor_indices = []
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
        neighbor_indices.append(idxs)
    
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        for j, idx in enumerate(neighbor_indices[i]):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping
    # For each word, which codewords cover it?
    # This is useful for finding replacements.
    # Instead of storing a list for each word (memory heavy for 2187 * 186),
    # we can just use the neighbors_arr for the candidate check.
    
    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)
            
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
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            # Phase 1: Try to remove 1, add 0 (size reduction)
            for i in indices:
                if time.time() >= deadline:
                    break
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                if not np.any(counts == 0):
                    code.pop(i)
                    improved = True
                    break
                else:
                    counts[neighbors_arr[c_remove, :]] += 1
            
            if improved:
                continue

            # Phase 2: Try to remove 2, add 1 (size reduction)
            # This is more expensive. We try pairs.
            # To keep it fast, we only try a subset of pairs or use a heuristic.
            # If we remove c_i and c_j, we have holes H. We need c_k covering H.
            # H = {w | count[w] == 1 and w in N(c_i) or N(c_j)}? No.
            # H = {w | count[w] becomes 0}.
            
            # Optimization: Only try pairs if the code is "tight".
            # For now, we implement a simple version:
            # Pick a random c_i. Remove it. Find holes H.
            # Find candidates c_k that cover H.
            # If a candidate c_k exists, and removing c_k (which is in code) and adding c_k... wait.
            # We want to replace {c_i, c_j} with {c_k}.
            # So we remove c_i and c_j. Holes H.
            # We need c_k in code \ {c_i, c_j} such that N(c_k) covers H.
            
            # Let's try a simpler "swap" first: remove 1, add 1 (size same, but different structure)
            # This helps escape local minima.
            
            for i in indices:
                if time.time() >= deadline:
                    break
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    # Find critical words
                    critical_mask = (counts == 0)
                    critical_words = np.where(critical_mask)[0]
                    
                    # Find candidates that cover all critical words
                    # A candidate c covers w if w in neighbors_arr[c]
                    # We can check: for each candidate c in code (excluding c_remove), 
                    # does neighbors_arr[c] cover all critical_words?
                    
                    # Efficient check:
                    # For each critical word w, we need a codeword in the set S = {c | w in N(c)}.
                    # The intersection of S for all w in critical_words must be non-empty.
                    
                    # Since cover_count is small, we can precompute for each word the list of codewords covering it.
                    # But we don't have that precomputed for the current code.
                    # We can compute it on the fly for the critical words.
                    
                    # For each critical word w, find which codewords in code cover it.
                    # code covers w if counts_before[w] > 0? No, counts_before[w] was 1 (since it became 0).
                    # So exactly one codeword covered w. That codeword is c_remove.
                    # Wait, if counts[w] became 0, it means c_remove was the ONLY codeword covering w.
                    # So we need a new codeword c_new that covers w.
                    # And it must cover ALL such w.
                    
                    # So we need c_new such that for all w in critical_words, w in N(c_new).
                    
                    # Let's iterate through code and check.
                    found_swap = False
                    for j, c_candidate in enumerate(code):
                        if j == i:
                            continue
                        # Check if c_candidate covers all critical_words
                        # critical_words is an array of indices.
                        # We need to check if all these indices are in neighbors_arr[c_candidate].
                        # neighbors_arr[c_candidate] is an array of size cover_count.
                        # We can use np.isin or set intersection.
                        # Set intersection is faster for small sets.
                        # But critical_words can be large.
                        # Let's use a mask.
                        # Create a boolean mask of size total_words, set critical_words to True.
                        # Then check if neighbors_arr[c_candidate] has any False in those positions?
                        # No, we need ALL critical_words to be in N(c_candidate).
                        
                        # Efficient way:
                        # For each critical word w, check if w is in neighbors_arr[c_candidate].
                        # This is O(|critical_words| * cover_count).
                        
                        # Alternative:
                        # Precompute a bitset or boolean array for each codeword? No, too much memory.
                        
                        # Let's just check.
                        all_covered = True
                        for w in critical_words:
                            if not np.any(neighbors_arr[c_candidate] == w):
                                all_covered = False
                                break
                        if all_covered:
                            # Perform swap
                            counts[neighbors_arr[c_candidate, :]] += 1
                            code[i] = c_candidate
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                    else:
                        counts[neighbors_arr[c_remove, :]] += 1
                else:
                    # This case should have been handled in Phase 1, but just in case
                    code.pop(i)
                    improved = True
                    break
        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
