import itertools
import time
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute neighbors
    neighbors = {}
    for idx in range(q**n):
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        w = tuple(w)
        nbrs = [idx]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nw = tuple(nw)
                    # Calculate index
                    nidx = 0
                    for j in range(n):
                        nidx = nidx * q + nw[j]
                    nbrs.append(nidx)
        neighbors[idx] = nbrs

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        return tuple(w)

    # Initial code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    code_set = set(initial_code)
    cover_count = [0] * (q**n)
    
    for c in code_set:
        c_idx = word_to_idx(c)
        for nbr in neighbors[c_idx]:
            cover_count[nbr] += 1

    start_time = time.time()
    deadline = start_time + 110
    
    # Heuristic: Try to remove words. If stuck, perform a swap to break structure.
    # Since direct removal fails on the perfect product code, we must perturb it.
    
    # Strategy: 
    # 1. Try to remove a word.
    # 2. If no word can be removed, pick a random word, remove it, 
    #    then add back a word that covers the most newly uncovered words (or just add a random one to restore coverage).
    #    Actually, if we remove a word, some words become uncovered.
    #    We must add a word to cover them.
    #    We want the net change in size to be 0 or -1.
    #    If we remove 1 and add 1, size is same.
    #    If we remove 1 and add 0 (if possible), size decreases.
    
    # Let's try: 
    # Pick a word c. Remove it.
    # Find uncovered words U.
    # If U is empty, we improved!
    # If U is not empty, find a word c' (not in code) that covers the most words in U.
    # If c' covers all U, and c' is not c, we swap c with c'. Size same.
    # But maybe c' also covers other words, allowing future removals?
    
    # Simpler: Just try to remove. If fail, swap two words to random positions?
    
    # Let's try a different initial code: Random 500 words?
    # No, might not cover.
    
    # Let's stick to the perturbation of the 729 code.
    
    code_list = list(code_set)
    
    while time.time() < deadline:
        removed_any = False
        # Try to remove
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            c_idx = word_to_idx(c)
            removable = True
            for nbr in neighbors[c_idx]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                for nbr in neighbors[c_idx]:
                    cover_count[nbr] -= 1
                removed_any = True
                break # Restart scan after removal
        
        if removed_any:
            continue
            
        # If no removal possible, try a swap to escape local minimum
        # Pick a random word, remove it, and add a random word from the space not in code.
        # This is a "kick".
        
        if len(code_list) > 450: # Only kick if we are not already very small
            idx_c = random.randint(0, len(code_list)-1)
            c = code_list[idx_c]
            c_idx = word_to_idx(c)
            
            # Temporarily remove
            for nbr in neighbors[c_idx]:
                cover_count[nbr] -= 1
            
            # Find uncovered
            uncovered = [nbr for nbr in neighbors[c_idx] if cover_count[nbr] == 0]
            
            if not uncovered:
                # We removed a word and didn't break coverage! 
                # (This shouldn't happen if removal check was thorough, but maybe due to logic gap)
                code_list.pop(idx_c)
                continue
            
            # Find a replacement word that covers uncovered words
            # We look for a word c' that covers all uncovered words.
            # This is a set cover problem on the small set 'uncovered'.
            # 'uncovered' size is at most 17.
            
            # Candidates: Any word in the space.
            # But we can restrict to words close to 'uncovered'.
            
            found_swap = False
            # Try all words in the space? 6561.
            # For each, check if it covers all uncovered.
            # This is 6561 * 17 checks. ~100k ops. Fast.
            
            best_c_prime = None
            best_score = -1
            
            for idx_cp in range(q**n):
                if idx_cp == c_idx:
                    continue
                # Check if idx_cp is already in code? 
                # We can add it, then remove c.
                # We want to minimize size, so swap is 0 change.
                # But maybe it enables future removals.
                
                # Count how many uncovered words it covers
                cov_count = 0
                cp_nbrs = neighbors[idx_cp]
                for u in uncovered:
                    if u in cp_nbrs:
                        cov_count += 1
                if cov_count == len(uncovered):
                    # It covers all uncovered
                    # Check if it adds redundancy elsewhere?
                    # Just pick it.
                    best_c_prime = idx_cp
                    break # Pick first one
            
            if best_c_prime is not None:
                # Perform swap
                code_list[idx_c] = idx_to_word(best_c_prime)
                # Update cover counts
                for nbr in neighbors[best_c_prime]:
                    cover_count[nbr] += 1
            else:
                # Could not find a single word to cover all uncovered.
                # Revert the removal
                for nbr in neighbors[c_idx]:
                    cover_count[nbr] += 1
                # Try another word
                continue

    return [list(c) for c in code_list]
