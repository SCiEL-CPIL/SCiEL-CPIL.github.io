import time
import random
import itertools
import sys

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    return _fallback_construct(q, n, radius)

def _fallback_construct(q, n, radius):
    # Simple fallback for non-standard cases
    if radius == 0:
        return [[i] * n for i in range(q)]
    # Greedy set cover for small instances
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        neighbors.append(nbrs)
    
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in neighbors[i]):
                continue
            cnt = sum(1 for j in neighbors[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in neighbors[best_idx]:
            covered[j] = True
    
    return codewords

def _construct_ternary_8_1():
    q = 3
    n = 8
    total = q ** n  # 6561
    deadline = time.time() + 140  # 10s safety margin

    def idx_to_digits(idx):
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    # Build neighbor list
    neighbors = [None] * total
    for idx in range(total):
        digits = idx_to_digits(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val != orig:
                    digits[pos] = val
                    nbrs.append(digits_to_idx(digits))
            digits[pos] = orig
        neighbors[idx] = nbrs
    
    # Precompute neighbor sets for fast intersection/coverage checks if needed, 
    # but lists are fine for counting.

    # Initial Code: Tetracode (9 words) x All (81 words) = 729 words
    # Tetracode: (a, b, a+b, a+2b) mod 3
    tetracode = []
    for a in range(3):
        for b in range(3):
            t = (a, b, (a + b) % 3, (a + 2 * b) % 3)
            # Convert first 4 digits to index
            idx_part = ((t[0] * 3 + t[1]) * 3 + t[2]) * 3 + t[3]
            tetracode.append(idx_part)
            
    # The full index is idx_part * 3^4 + tail_idx
    # tail_idx ranges 0..80
    initial_code = []
    for tc in tetracode:
        for tail in range(81):
            initial_code.append(tc * 81 + tail)
            
    code_set = set(initial_code)
    code_list = list(initial_code)
    
    # Initialize coverage counts
    cover_count = [0] * total
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
            
    best_code = list(code_list)
    best_size = len(code_list)
    
    # Optimization Loop
    # We use a "Shake and Prune" strategy.
    # 1. Try to remove a random codeword.
    # 2. If it causes uncovered words, find a replacement that covers them.
    # 3. After any change, try to prune redundant codewords.
    
    rng = random.Random(42)
    
    def try_prune(code_list, cover_count, code_set):
        """Try to remove redundant codewords. Returns number removed."""
        removed = 0
        # Shuffle to break symmetry
        rng.shuffle(code_list)
        i = 0
        while i < len(code_list):
            c = code_list[i]
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                # Remove c
                code_list.pop(i)
                code_set.remove(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed += 1
                # Do not increment i, check the new element at position i
            else:
                i += 1
        return removed

    # Initial prune
    try_prune(code_list, cover_count, code_set)
    if len(code_list) < best_size:
        best_size = len(code_list)
        best_code = list(code_list)

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if not code_list:
            break
            
        # Pick a random codeword to remove
        idx = rng.randint(0, len(code_list) - 1)
        c_remove = code_list[idx]
        
        # Temporarily remove
        for nbr in neighbors[c_remove]:
            cover_count[nbr] -= 1
        code_list.pop(idx)
        code_set.remove(c_remove)
        
        # Check for uncovered words
        uncovered = []
        for nbr in neighbors[c_remove]:
            if cover_count[nbr] == 0:
                uncovered.append(nbr)
                
        if not uncovered:
            # Successfully removed, size decreased
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            # Try to prune more
            try_prune(code_list, cover_count, code_set)
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            continue
            
        # If uncovered, we need to add a codeword that covers all of them
        # Candidates are neighbors of uncovered words
        candidates = set()
        for u in uncovered:
            candidates.update(neighbors[u])
            
        # Remove c_remove from candidates (we already removed it, but just in case)
        candidates.discard(c_remove)
        
        # Try to find a candidate that covers all uncovered words
        # And ideally, is not already in the code (if it is, we are just swapping)
        # If it is in the code, we are effectively swapping c_remove with an existing word? 
        # No, if c_new is in code_set, we just add it back? No, c_remove is already removed.
        # If c_new is in code_set, adding it increases size by 1 (net 0 change if we put c_remove back? No).
        # We want to keep size same or decrease. 
        # If we add c_new, size increases by 1. 
        # We can only accept this if it allows us to prune others later.
        
        found_repair = False
        # Sort candidates: prefer those NOT in code_set? 
        # Actually, if c_new is in code_set, we are duplicating? No, code_set is a set.
        # If c_new is in code_set, it's already there. We just need to add it to code_list?
        # No, if it's in code_set, it's already counted in cover_count.
        # Wait, we removed c_remove. c_new is in code_set, so its neighbors are still covered by it.
        # If c_new covers `uncovered`, then `uncovered` is not actually uncovered?
        # Let's re-evaluate.
        # `uncovered` are words with cover_count == 0.
        # If c_new is in code_set, its neighbors have cover_count >= 1 (from c_new).
        # So c_new CANNOT cover any word in `uncovered` if c_new is in code_set?
        # No, that's wrong. `cover_count` was decremented for c_remove.
        # If c_new is in code_set, it is still active. Its neighbors have cover_count >= 1.
        # So if a word u is in `uncovered`, it means NO active codeword covers it.
        # If c_new is in code_set and covers u, then u would have cover_count >= 1.
        # Contradiction.
        # Therefore, ANY codeword that covers `uncovered` MUST be NOT in code_set.
        
        for c_new in candidates:
            if c_new in code_set:
                continue
                
            # Check if c_new covers all uncovered
            valid_repair = True
            for u in uncovered:
                if cover_count[u] == 0: # It is uncovered
                    # Check if c_new covers u
                    # We can check if u is in neighbors[c_new]
                    # Since neighbors are lists, this is O(17)
                    if u not in neighbors[c_new]:
                        valid_repair = False
                        break
            if valid_repair:
                # Perform the add
                code_list.append(c_new)
                code_set.add(c_new)
                for nbr in neighbors[c_new]:
                    cover_count[nbr] += 1
                found_repair = True
                break
                
        if not found_repair:
            # Cannot repair with a single word. Restore c_remove.
            code_list.append(c_remove)
            code_set.add(c_remove)
            for nbr in neighbors[c_remove]:
                cover_count[nbr] += 1
            continue
            
        # If we found a repair, we have a valid code of same size.
        # Now try to prune to see if we can reduce size.
        try_prune(code_list, cover_count, code_set)
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)
            
        # Occasionally, do a more aggressive shake: remove 2, add 1? 
        # Or just stick to 1-for-1 swaps + prune.
        
    return [idx_to_digits(c) for c in best_code]
