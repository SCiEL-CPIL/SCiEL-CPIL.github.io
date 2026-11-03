import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 120s limit, stop at 115s
    deadline = start_time + 115.0

    total_words = q ** n
    
    # Precompute index to tuple and tuple to index
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
    # neighbors_arr[i] contains indices of all words within Hamming distance <= radius of word i
    # For radius 1, size is 1 + n*(q-1)
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
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
        neighbors_arr[i, :len(idxs)] = idxs
        
    # Precompute reverse mapping: for each word, which codewords cover it
    # reverse_cover[word] = list of codeword indices that cover this word
    # Since the graph is undirected and regular, reverse_cover is essentially the same as neighbors
    # But let's build it explicitly to be safe, or just use neighbors_arr.T?
    # neighbors_arr is (total_words, cover_count).
    # reverse_cover[w] needs to be a list of indices i such that w is in neighbors_arr[i].
    # We can just use neighbors_arr.T if we flatten it? No, variable lengths if radius > 1? 
    # For radius 1, it's regular. 
    # Let's just use neighbors_arr directly. If word w is covered by codeword c, then c is in neighbors_arr[w].
    # So reverse_cover[w] is just neighbors_arr[w].
    # Wait, is it? 
    # dist(c, w) <= 1 iff dist(w, c) <= 1. Yes.
    # So reverse_cover[w] is simply neighbors_arr[w].
    
    def greedy_construct_vectorized(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            if time.time() > deadline - 5:
                break
            # Vectorized gain computation
            # uncovered[neighbors_arr] gives a boolean array of shape (total_words, cover_count)
            # Summing along axis 1 gives the number of uncovered words covered by each candidate
            gains = uncovered[neighbors_arr].sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            # Random tie-breaking
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            if len(best_candidates) > 1:
                chosen = random.choice(best_candidates.tolist())
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
        """Remove redundant codewords."""
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            # Iterate in reverse to allow popping? Or just build new list.
            # Building new list is safer.
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    # Cannot remove
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    # Can remove
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, time_limit):
        """
        Local search with 1-for-1 swaps.
        """
        code = list(code)
        if not code:
            return code

        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        iterations = 0
        while improved and time.time() < time_limit:
            improved = False
            iterations += 1
            # Limit iterations to prevent infinite loops if stuck
            if iterations > 1000:
                break
                
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= time_limit:
                    break
                
                c_remove = code[i]
                # Tentatively remove
                counts[neighbors_arr[c_remove, :]] -= 1
                
                # Check if any word is now uncovered
                # We only need to check the words that c_remove covered
                # But it's faster to just check global min? 
                # No, counts is global. 
                # If any count is 0, we have a problem.
                # We can check if min(counts) == 0.
                # But min is O(N). 
                # Better: check the specific words affected.
                # affected_words = neighbors_arr[c_remove]
                # if np.any(counts[affected_words] == 0): ...
                
                affected_words = neighbors_arr[c_remove, :]
                zero_mask = (counts[affected_words] == 0)
                
                if not np.any(zero_mask):
                    # Safe to remove
                    # counts already updated (subtracted)
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Need to find a replacement
                    # critical_words are those in affected_words where count is 0
                    critical_words = affected_words[zero_mask]
                    
                    # Find candidates that cover ALL critical words
                    # Candidate c_new must be in reverse_cover[cw] for all cw in critical_words
                    # reverse_cover[cw] is neighbors_arr[cw]
                    
                    # Start with the set of candidates covering the first critical word
                    candidate_set = set(neighbors_arr[critical_words[0]].tolist())
                    
                    for cw in critical_words[1:]:
                        if not candidate_set:
                            break
                        candidate_set.intersection_update(neighbors_arr[cw].tolist())
                    
                    # Remove c_remove itself from candidates
                    candidate_set.discard(c_remove)
                    
                    if candidate_set:
                        # Pick a random candidate
                        best_w = random.choice(list(candidate_set))
                        
                        # Swap c_remove with best_w
                        # First, add best_w
                        counts[neighbors_arr[best_w, :]] += 1
                        # c_remove is already removed from counts
                        code[i] = best_w
                        improved = True
                        break
                    else:
                        # Cannot find replacement, revert
                        counts[neighbors_arr[c_remove, :]] += 1
                        continue
        
        return code

    best_size = float('inf')
    best_code = []
    
    seed = 0
    # Run multiple restarts
    while time.time() < deadline - 10:
        # 1. Greedy Construction
        code = greedy_construct_vectorized(seed)
        
        # 2. Reduction
        code = reduce_code_fast(code)
        
        # 3. Local Search
        if time.time() < deadline - 5:
            code = local_search_improve(code, deadline - 5)
            # Final reduction after local search
            code = reduce_code_fast(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        seed += 1
        
    # Convert best_code indices to tuples
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
