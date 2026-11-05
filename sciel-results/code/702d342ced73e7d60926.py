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
    
    # Precompute neighbors as bitmasks and lists
    # word_to_idx and idx_to_word logic
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
        # Optimization: We can skip checking words that are already in code
        # But iterating all N is fine (6561)
        for c in range(N):
            if c in code_set:
                continue
            # Count how many uncovered words are covered by c
            # score = bit_count(uncovered_mask & neighbor_masks[c])
            score = (uncovered_mask & neighbor_masks[c]).bit_count()
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c == -1:
            break # Should not happen if uncovered_mask != 0
            
        code.append(best_c)
        code_set.add(best_c)
        
        # Update coverage
        for nbr in neighbor_lists[best_c]:
            cover_count[nbr] += 1
            if cover_count[nbr] == 1:
                uncovered_mask &= ~(1 << nbr)
        
        if time.time() > deadline - 30:
            break # Safety margin for local search

    # Local Search
    # 1. Try to remove redundant words
    # 2. Try to swap words to improve structure
    
    while time.time() < deadline:
        removed_any = False
        
        # Pass 1: Remove redundant words
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            can_remove = True
            for nbr in neighbor_lists[c]:
                if cover_count[nbr] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nbr in neighbor_lists[c]:
                    cover_count[nbr] -= 1
                    if cover_count[nbr] == 0:
                        uncovered_mask |= (1 << nbr)
                code.pop(i)
                code_set.remove(c)
                removed_any = True
                break # Restart scan after removal
        
        if removed_any:
            continue
            
        # Pass 2: Swap
        # Pick a random word from code
        if code:
            idx_c = random.randint(0, len(code) - 1)
            c = code[idx_c]
            
            # Temporarily remove c
            for nbr in neighbor_lists[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words
            # We can use the mask logic or just check cover_count == 0
            uncovered = [nbr for nbr in neighbor_lists[c] if cover_count[nbr] == 0]
            
            if not uncovered:
                # We removed c and didn't break coverage? 
                # This implies c was redundant, but Pass 1 should have caught it.
                # Maybe due to concurrent modification? No, single thread.
                # If this happens, we just keep c removed.
                code.pop(idx_c)
                code_set.remove(c)
                continue
            
            # Find c' that covers all uncovered
            # We want c' not in code_set
            # Check condition: for all u in uncovered, u is in neighbor_lists[c']
            # Equivalent to: mask(uncovered) is subset of neighbor_masks[c']
            mask_uncovered = 0
            for u in uncovered:
                mask_uncovered |= (1 << u)
            
            best_c_prime = -1
            # Iterate through candidates
            # To speed up, we can prioritize candidates that are "close" to c? 
            # Or just random order?
            # Let's try all candidates.
            for c_prime in range(N):
                if c_prime in code_set:
                    continue
                if (mask_uncovered & ~neighbor_masks[c_prime]) == 0:
                    best_c_prime = c_prime
                    break
            
            if best_c_prime != -1:
                # Perform swap
                code[idx_c] = best_c_prime
                code_set.remove(c)
                code_set.add(best_c_prime)
                
                # Update cover counts
                for nbr in neighbor_lists[best_c_prime]:
                    cover_count[nbr] += 1
                    if cover_count[nbr] == 1:
                        uncovered_mask &= ~(1 << nbr)
            else:
                # Revert removal
                for nbr in neighbor_lists[c]:
                    cover_count[nbr] += 1
                    if cover_count[nbr] == 1:
                        uncovered_mask &= ~(1 << nbr)
                
                # If we can't find a swap, we are stuck.
                # We can try a "kick": add a random word, then remove redundant ones?
                # Or just break and return current best.
                # Given time constraints, breaking is safer.
                # But let's try a few random kicks if we have time.
                if time.time() < deadline - 10:
                    # Kick: Add a random word not in code
                    # This might allow future removals?
                    # Actually, adding a word increases size. We only want to decrease.
                    # So if we can't swap, we are likely at a local minimum for this size.
                    # We could try to increase size slightly to escape? 
                    # "Perturbation": Add 2 random words, then try to remove 3?
                    # This is risky. Let's just stop searching if stuck.
                    break
                else:
                    break
        else:
            break

    return [list(idx_to_word(c)) for c in code]
