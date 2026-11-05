import numpy as np
import itertools

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    neighbors = np.empty((N, sphere_size), dtype=int)
    
    # Precompute neighbors
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    covered = np.zeros(N, dtype=bool)
    cover_count = np.zeros(N, dtype=int)
    C = []
    
    # 1. Greedy Phase
    while covered.sum() < N:
        # Calculate gain for each candidate
        # Gain is the number of currently uncovered neighbors
        # We can compute this as: sum(~covered[neighbors[i]])
        # To speed this up, we can maintain a 'uncovered_neighbors_count' array?
        # Or just use the vectorized operation. N=2187, sphere=20. 2187*20 = 43k ops.
        # Doing this 180 times is ~7.8M ops. Very fast.
        gains = (~covered[neighbors]).sum(axis=1)
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        C.append(best_idx)
        cover_count[neighbors[best_idx]] += 1
        covered[neighbors[best_idx]] = True
        
    # 2. Local Search Phase: 1-Exchange
    # Try to swap a codeword c with a non-codeword c' such that the cover remains valid.
    # If valid, we might reduce the "cost" (e.g. number of critical words) or just accept it if it allows further removals?
    # Actually, if we swap, size stays same. We need to enable removal.
    # Strategy: 
    # 1. Try to remove any redundant codeword.
    # 2. If no redundant, try to swap c with c' to increase redundancy (reduce critical count).
    
    in_C = np.zeros(N, dtype=bool)
    for c in C:
        in_C[c] = True
        
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Step A: Remove redundant codewords
        removed = False
        for i in range(len(C)):
            c = C[i]
            nb = neighbors[c]
            cover_count[nb] -= 1
            
            if np.any(cover_count[nb] == 0):
                # Restoration
                cover_count[nb] += 1
            else:
                C.pop(i)
                in_C[c] = False
                removed = True
                break
        
        if removed:
            improved = True
            continue
            
        # Step B: Swap to improve structure
        # Calculate current critical count
        critical_mask = (cover_count == 1)
        current_critical = np.sum(critical_mask)
        
        best_swap = None
        best_score = current_critical
        best_delta = 0
        
        # For each codeword c, try swapping with each non-codeword c'
        for c in C:
            nb_c = neighbors[c]
            
            # Words that are critical and covered by c
            # If we remove c, these words lose their only cover.
            # We need c' to cover them, OR we need to accept they become uncovered (invalid).
            # So c' MUST cover all words in nb_c that are critical.
            critical_in_c = nb_c[critical_mask[nb_c]]
            
            if len(critical_in_c) == 0:
                # c does not cover any critical words.
                # Removing c makes no word uncovered.
                # We can swap with any c' that doesn't break coverage?
                # Actually, if we remove c, all words in nb_c still have count >= 1.
                # So we can add any c' and the cover is valid.
                # We should pick a c' that maximizes redundancy gain?
                # Gain in redundancy = (count before - count after) for words in nb_c' that are not in nb_c?
                # Simpler: Pick a c' that minimizes the new critical count.
                # New critical count = current_critical - (critical words in nb_c that become non-critical) + (new critical words in nb_c')
                # Since critical_in_c is empty, no critical words are affected by removal.
                # Adding c' increases count in nb_c'.
                # Words in nb_c' with count 0? No, covered is full.
                # Words in nb_c' with count 1 become 2 (good).
                # Words in nb_c' with count > 1 stay > 1.
                # So adding c' never creates new critical words, it only reduces them.
                # Reduction = number of words in nb_c' with count 1.
                # We want to maximize this reduction.
                candidates = np.where(~in_C)[0]
                if len(candidates) == 0: continue
                
                # Vectorized gain calculation for all candidates?
                # gain[c'] = sum(cover_count[neighbors[c']] == 1)
                # This is expensive to do for all candidates at once.
                # But candidates ~ 2000. neighbors size 20.
                # We can just iterate? 2000 iterations is fine.
                best_c_prime = -1
                max_reduction = 0
                for cp in candidates:
                    nb_cp = neighbors[cp]
                    reduction = np.sum(cover_count[nb_cp] == 1)
                    if reduction > max_reduction:
                        max_reduction = reduction
                        best_c_prime = cp
                
                if best_c_prime != -1 and max_reduction > 0:
                    new_critical = current_critical - max_reduction
                    if new_critical < best_score:
                        best_score = new_critical
                        best_swap = (c, best_c_prime)
            else:
                # c covers some critical words.
                # c' MUST cover all these critical words.
                # Find candidates cp such that critical_in_c is a subset of neighbors[cp].
                # We can filter candidates.
                candidates = np.where(~in_C)[0]
                if len(candidates) == 0: continue
                
                # Filter candidates
                # For each candidate, check if it covers all critical_in_c
                # We can use set intersection logic or boolean masks.
                # Let's just iterate, it's small enough.
                best_c_prime = -1
                best_new_critical = current_critical + 100 # init high
                
                for cp in candidates:
                    nb_cp = neighbors[cp]
                    # Check if all critical_in_c are in nb_cp
                    # Create a mask for critical_in_c
                    # If any critical word is NOT in nb_cp, skip.
                    # np.isin(critical_in_c, nb_cp) should be all True.
                    if not np.all(np.isin(critical_in_c, nb_cp)):
                        continue
                    
                    # Calculate new critical count
                    # Remove c: cover_count[nb_c] -= 1
                    # Add cp: cover_count[nb_cp] += 1
                    # We can simulate this locally.
                    # Words in nb_c:
                    #   - If count was 1 (critical), becomes 0 (uncovered) -> INVALID unless covered by cp.
                    #     We already checked cp covers all critical in nb_c.
                    #     So they become 1 (covered by cp only).
                    #   - If count was > 1, becomes > 0.
                    #     If it becomes 1, it becomes critical.
                    # Words in nb_cp:
                    #   - If count was 0 (impossible, covered is full), becomes 1.
                    #   - If count was 1, becomes 2 (not critical).
                    #   - If count was > 1, becomes > 1.
                    #   - If word is in both nb_c and nb_cp:
                    #     Net change is 0.
                    
                    # Let's compute delta carefully.
                    # Let A = nb_c, B = nb_cp.
                    # Let C0 = cover_count.
                    # New count C1 = C0 - 1_A + 1_B.
                    
                    # Critical words in new state: C1 == 1.
                    
                    # We can just compute the new critical count for this specific swap.
                    # Since N is small (2187), we can do a vectorized update on a copy?
                    # No, copying 2187 ints is fast.
                    temp_count = cover_count.copy()
                    temp_count[nb_c] -= 1
                    temp_count[nb_cp] += 1
                    
                    new_critical = np.sum(temp_count == 1)
                    
                    if new_critical < best_new_critical:
                        best_new_critical = new_critical
                        best_c_prime = cp

                if best_c_prime != -1:
                    if best_new_critical < best_score:
                        best_score = best_new_critical
                        best_swap = (c, best_c_prime)
                        
        if best_swap:
            c, cp = best_swap
            nb_c = neighbors[c]
            nb_cp = neighbors[cp]
            
            cover_count[nb_c] -= 1
            cover_count[nb_cp] += 1
            
            C.remove(c)
            C.append(cp)
            in_C[c] = False
            in_C[cp] = True
            improved = True

    result_words = []
    for idx in C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
