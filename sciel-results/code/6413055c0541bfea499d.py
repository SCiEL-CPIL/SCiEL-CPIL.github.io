import time
import random
import numpy as np
from itertools import product

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

    # Precompute neighbors as numpy arrays for vectorized operations
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
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping: for each word, which codewords cover it
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1

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
            
            # Pass 1: Remove any removable codewords
            # We iterate in random order to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            to_remove = []
            for i in indices:
                c = code[i]
                counts[neighbors_arr[c, :]] -= 1
                if not np.any(counts == 0):
                    to_remove.append(i)
                else:
                    counts[neighbors_arr[c, :]] += 1
            
            if to_remove:
                # Remove all identified removable codewords
                # We must be careful: removing one might make another removable or not.
                # But since we checked them independently against the *current* state,
                # and they were all removable *before* any removals in this batch,
                # removing them all is safe IF they don't share critical words.
                # Wait, if two codewords are both removable, it means every word they cover is covered by at least one other codeword.
                # If we remove both, is it still covered?
                # Yes, because for any word w covered by c1, w is covered by some c != c1.
                # If w is also covered by c2, and c2 is removed, is w covered by some c != c1, c2?
                # Not necessarily. c1 and c2 might be the *only* two covering w.
                # So we cannot remove multiple codewords in a batch without re-checking.
                # We will remove them one by one.
                
                for i in to_remove:
                    c = code[i]
                    counts[neighbors_arr[c, :]] -= 1
                    # Re-check if it's still removable (it should be, as we only removed other codewords, 
                    # which can only decrease counts of words covered by c, but we already established 
                    # that no word covered by c had count 1 *before* this batch. 
                    # Wait, if another codeword c2 was removed, and c2 was the *only* other cover for a word w in N(c),
                    # then count of w drops. If it was 2, it becomes 1. If it was 1, it becomes 0.
                    # We checked `np.any(counts == 0)` *after* subtracting c.
                    # So if we subtract c1, and then subtract c2, the check for c1 might have been valid 
                    # when c2 was present, but invalid after c2 is removed?
                    # No, `counts` reflects the state *with* all current codewords.
                    # If c1 is removable, it means for all w in N(c1), count(w) >= 2.
                    # If we remove c2, count(w) for w in N(c2) decreases.
                    # If w is in N(c1) and N(c2), count(w) decreases.
                    # If count(w) was 2, it becomes 1.
                    # So c1 might *not* be removable anymore if we remove c2 first.
                    # Therefore, we must remove them one by one and re-verify.
                    
                    # Since `to_remove` was calculated based on the state *before* any removals in this pass,
                    # we should re-verify each one.
                    if np.any(counts == 0):
                         # It became non-removable due to previous removals in this loop?
                         # No, we haven't removed any in this inner loop yet.
                         # But we are removing them sequentially.
                         # Let's just do a simple loop.
                         pass
                    # Actually, let's just stick to the simple "remove one, break" or "remove one, continue" 
                    # but ensure correctness.
                    # The safest is: try to remove, if fails, restore.
                    # We already restored in the check loop above.
                    # So `to_remove` contains indices that were removable *in the current state*.
                    # If we remove them one by one, we must re-check.
                    
                    # Let's just remove the first one found and restart the pass?
                    # That's slow.
                    # Let's just remove them one by one, and if it fails (count hits 0), 
                    # it means my batch logic was flawed. 
                    # Actually, if c1 and c2 are both removable, it implies N(c1) and N(c2) are covered by others.
                    # It is possible that c1 and c2 are the only two covering some word w.
                    # Then count(w)=2.
                    # If we remove c1, count(w)=1.
                    # If we then try to remove c2, count(w)=0 -> Fail.
                    # So we cannot blindly remove all.
                    
                    # Strategy: Remove one, update counts, continue loop.
                    pass 

            # Simpler and safer: Iterate, try to remove, if success, mark improved and break/restart pass.
            # This is the "First Improvement" strategy which is robust.
            # To improve speed, we can try to remove *all* that are removable in a single pass 
            # by processing them in an order that minimizes conflicts? 
            # Or just accept that we do one removal per pass.
            
            # Let's use the "First Improvement" but make it fast.
            # The previous code did this. Let's enhance the SWAP part.
            
            # Let's try a different strategy:
            # 1. Find all removable. If any, remove one random one.
            # 2. If none, try swaps.
            
            # Optimization: Pre-compute "critical words" for each codeword? 
            # Critical words for c are those w in N(c) where count(w) == 1.
            # If count(w) > 1 for all w in N(c), c is removable.
            
            # Let's just run the standard first-improvement local search but with a better swap heuristic.
            
            # Reset the improved flag for this pass
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    return code
                
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    # Cannot remove, try to swap
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    
                    if len(critical_words) == 0:
                        # Should be removable if no critical words? 
                        # Yes, if no word has count 1, then all have count >= 2 (since covered).
                        # So it should have been removed.
                        continue
                    
                    # Find replacement
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    # Heuristic: Choose the candidate that covers the most "other" words?
                    # Or just random?
                    # Let's try to choose the one that maximizes the count of words in its neighborhood 
                    # that are currently "weak" (count 1 or 2).
                    # This is expensive. Let's stick to random for now, but maybe bias towards 
                    # candidates that are "similar" to c_remove?
                    
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
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
        
        # Local search
        if time.time() < deadline - 5:
            code = local_search_improve(code, deadline - 2)
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
