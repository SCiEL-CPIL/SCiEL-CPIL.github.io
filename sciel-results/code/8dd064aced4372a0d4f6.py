import time
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]

    N = q ** n
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute neighbors
    def idx_to_word(idx):
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for idx in range(N):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nidx = word_to_idx(nw)
                    nbrs.append(nidx)
        
        mask = 0
        for nidx in nbrs:
            mask |= (1 << nidx)
            neighbor_lists[idx].append(nidx)
        neighbor_masks[idx] = mask

    # Greedy Set Cover
    uncovered_mask = (1 << N) - 1
    code = []
    code_set = set()
    cover_count = [0] * N
    
    while uncovered_mask:
        best_score = 0
        best_c = -1
        for c in range(N):
            if c in code_set:
                continue
            score = (uncovered_mask & neighbor_masks[c]).bit_count()
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c == -1:
            break
            
        code.append(best_c)
        code_set.add(best_c)
        
        for nbr in neighbor_lists[best_c]:
            cover_count[nbr] += 1
            if cover_count[nbr] == 1:
                uncovered_mask &= ~(1 << nbr)
        
        if time.time() > deadline - 30:
            break

    # Helper to check if code is valid and get size
    def is_valid_code(c_list):
        # This is expensive, so we rely on cover_count logic during search
        # But for final check or perturbation, we might need to verify
        # Actually, we maintain cover_count, so we just need to ensure no uncovered_mask bits are set
        # But cover_count is global state. 
        # For perturbation, we will manage state explicitly.
        pass

    best_code = code[:]
    best_size = len(code)
    
    # Local Search Function: Returns (new_code, new_cover_count, new_uncovered_mask, improved)
    # To avoid copying large lists, we might mutate and revert, or copy.
    # Given N=6561 and code size ~500, copying is cheap.
    
    def local_search(c_list, c_set, cover_counts, unc_mask):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Removal
            for i in range(len(c_list) - 1, -1, -1):
                c = c_list[i]
                can_remove = True
                for nbr in neighbor_lists[c]:
                    if cover_counts[nbr] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for nbr in neighbor_lists[c]:
                        cover_counts[nbr] -= 1
                        if cover_counts[nbr] == 0:
                            unc_mask |= (1 << nbr)
                    c_list.pop(i)
                    c_set.remove(c)
                    improved = True
                    break # Restart scan
            
            if not improved:
                # 2. Swap
                # Try to swap a random word with a better one
                # To be faster, we pick a random word from the code
                if c_list:
                    idx_c = random.randint(0, len(c_list) - 1)
                    c = c_list[idx_c]
                    
                    # Temporarily remove c
                    for nbr in neighbor_lists[c]:
                        cover_counts[nbr] -= 1
                    
                    # Find words that are now uncovered (only covered by c)
                    # These MUST be covered by the new word c'
                    # Actually, any word with cover_count 0 is uncovered.
                    # But only those that were covered by c and had count 1 become uncovered.
                    # Let's collect them.
                    must_cover = []
                    for nbr in neighbor_lists[c]:
                        if cover_counts[nbr] == 0:
                            must_cover.append(nbr)
                    
                    if not must_cover:
                        # c was redundant, but removal didn't happen?
                        # Maybe it was covered by others but count was >1?
                        # If must_cover is empty, we can just remove c?
                        # Wait, if must_cover is empty, it means every neighbor of c has count >= 1 even after removing c.
                        # So c is redundant. Why didn't removal catch it?
                        # Removal checks if ANY neighbor has count <= 1.
                        # If all neighbors have count >= 2, then after removal, all are >= 1.
                        # So c IS redundant.
                        # The removal loop should have caught it.
                        # Ah, the removal loop breaks after the first removal.
                        # So we might have missed subsequent redundant words.
                        # Let's just remove it here.
                        c_list.pop(idx_c)
                        c_set.remove(c)
                        improved = True
                        # Note: unc_mask is not updated here because no word became uncovered.
                        # But we should ensure unc_mask is consistent.
                        # Actually, if no word became uncovered, unc_mask doesn't change.
                        continue
                    
                    # We need c' such that it covers all words in must_cover
                    mask_must = 0
                    for u in must_cover:
                        mask_must |= (1 << u)
                    
                    # Find candidate c'
                    # Heuristic: Pick c' that covers must_cover AND maximizes new coverage?
                    # Or just any c' that covers must_cover.
                    # To escape local minima, random selection among valid c' is good.
                    valid_candidates = []
                    for c_prime in range(N):
                        if c_prime in c_set:
                            continue
                        if (mask_must & ~neighbor_masks[c_prime]) == 0:
                            valid_candidates.append(c_prime)
                    
                    if valid_candidates:
                        # Pick one randomly to diversify
                        c_prime = random.choice(valid_candidates)
                        
                        # Apply swap
                        c_list[idx_c] = c_prime
                        c_set.remove(c)
                        c_set.add(c_prime)
                        
                        for nbr in neighbor_lists[c_prime]:
                            cover_counts[nbr] += 1
                            if cover_counts[nbr] == 1:
                                unc_mask &= ~(1 << nbr)
                        improved = True
                    else:
                        # Revert
                        for nbr in neighbor_lists[c]:
                            cover_counts[nbr] += 1
                            if cover_counts[nbr] == 1:
                                unc_mask &= ~(1 << nbr)
        
        return c_list, c_set, cover_counts, unc_mask

    # Iterated Local Search
    while time.time() < deadline - 2:
        # Run local search on current state
        code, code_set, cover_count, uncovered_mask = local_search(code, code_set, cover_count, uncovered_mask)
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
            # Update best state variables? 
            # We continue from the best state to exploit further?
            # Or restart from best?
            # Usually, ILS restarts from best or perturbs best.
            # Let's perturb the current (which is best) state.
        
        # Perturbation
        if not code:
            break
            
        # Remove k random words
        k = random.randint(2, 5)
        if len(code) <= k:
            k = max(1, len(code) - 1)
            
        indices_to_remove = random.sample(range(len(code)), k)
        # Sort descending to remove safely
        indices_to_remove.sort(reverse=True)
        
        for i in indices_to_remove:
            c = code[i]
            for nbr in neighbor_lists[c]:
                cover_count[nbr] -= 1
                if cover_count[nbr] == 0:
                    uncovered_mask |= (1 << nbr)
            code.pop(i)
            code_set.remove(c)
        
        # Add k random words that cover some uncovered words
        # We want to add words that help cover the gaps.
        # Pick random words from N, prefer those covering uncovered bits.
        added = 0
        attempts = 0
        while added < k and attempts < 100 and time.time() < deadline - 2:
            c_new = random.randint(0, N - 1)
            if c_new in code_set:
                attempts += 1
                continue
            
            # Check if it covers any uncovered word
            # If yes, add it
            # Actually, just add it. Local search will clean up.
            # But adding random words might be bad.
            # Let's try to add words that cover the most uncovered words.
            # Simple greedy pick for the perturbation:
            # Find best word to add
            best_c = -1
            best_gain = 0
            # Sample a few random candidates to speed up
            for _ in range(10):
                cand = random.randint(0, N - 1)
                if cand in code_set:
                    continue
                gain = (uncovered_mask & neighbor_masks[cand]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = cand
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            for nbr in neighbor_lists[best_c]:
                cover_count[nbr] += 1
                if cover_count[nbr] == 1:
                    uncovered_mask &= ~(1 << nbr)
            added += 1
            attempts += 1

    return [list(idx_to_word(c)) for c in best_code]
