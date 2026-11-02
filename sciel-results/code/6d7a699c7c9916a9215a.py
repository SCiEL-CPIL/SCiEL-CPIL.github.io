"""Fresh design: Simulated Annealing with swap moves for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math
import itertools


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 108.0  # safety margin under 120s

    total = q ** n
    pos_step = [q ** (n - 1 - i) for i in range(n)]

    def word_to_idx(word):
        idx = 0
        for i, val in enumerate(word):
            idx += val * pos_step[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = (idx // pos_step[i]) % q
        return word

    def build_cover(codewords):
        """Return coverage count array."""
        cover = np.zeros(total, dtype=np.int32)
        for w in codewords:
            idx = word_to_idx(w)
            cover[idx] += 1
            if radius >= 1:
                for i in range(n):
                    cur_sym = w[i]
                    step = pos_step[i]
                    base = idx - cur_sym * step
                    for v in range(q):
                        if v != cur_sym:
                            cover[base + v * step] += 1
        return cover

    def is_valid_cover(cover):
        return cover.min() >= 1

    # Start with known good construction
    # For q=3, n=8, radius=1, the tetracode product is a strong start.
    # Tetracode is a [4,2,3] code over GF(3).
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Repeat tetracode structure? No, the product construction is:
        # If C1 is a code over GF(q) length n1 and C2 is a code over GF(q) length n2,
        # then C1 x C2 is a code over GF(q) length n1+n2.
        # However, the "tetracode" is often used in a different way for covering codes.
        # Let's stick to the seed from the previous program which was:
        # seed = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        # This is C_tetracode x {0..q-1}^(n-4).
        # This is valid because d(C x D) = min(d(C), d(D)) is not quite right for covering radius,
        # but K_q(n1+n2, r) <= K_q(n1, r1) * K_q(n2, r2) if r1+r2=r? No.
        # Actually, if C covers {0..q-1}^n1 with radius r1 and D covers {0..q-1}^n2 with radius r2,
        # then C x D covers {0..q-1}^(n1+n2) with radius r1+r2.
        # Here we want radius 1. If we use Tetracode (radius 1? No, Tetracode is perfect? No.)
        # Tetracode is a [4,2,3] code. Its covering radius is 1?
        # Distance 3 means min distance 3. Covering radius is 1.
        # So Tetracode covers GF(3)^4 with radius 1.
        # The set {0..q-1}^(n-4) covers GF(3)^(n-4) with radius 0.
        # So the product covers GF(3)^n with radius 1+0=1.
        # Size is 9 * 3^(n-4). For n=8, size is 9 * 81 = 729.
        # Wait, the previous program used this and got 729?
        # The task says best known is 486.
        # The previous program's score was 1.0, which is the seed.
        # The seed program likely returned this 729 size code?
        # Let's check the previous code's output size.
        # The previous code started with this seed and then did local search.
        # If the previous score was 1.0, and the seed is 729, then the best known in the population might be 729?
        # Or maybe the seed was different?
        # Let's assume the previous seed was the standard product construction.
        
        seed = []
        # Generate tetracode words
        t_words = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generate tails
        tails = list(itertools.product(range(q), repeat=n - 4))
        for h in t_words:
            for tail in tails:
                seed.append(h + list(tail))
    else:
        # Fallback: Random or simple grid
        seed = [[random.randrange(q) for _ in range(n)] for _ in range(100)]
        # Ensure validity? Random might not be valid.
        # For general q,n,r, this is hard.
        # But the task is specific: q=3, n=8, r=1.
        # We assume the evaluator passes these parameters.
        # If not, we return a large random set? No, must be valid.
        # Let's just use the product construction if possible, else brute force small?
        # Given the constraints, we assume q=3, n=8.
        pass

    best = list(seed)
    cover = build_cover(best)
    
    if not is_valid_cover(cover):
        # Fallback: Add random words until valid (expensive but safe)
        while not is_valid_cover(cover):
            w = [random.randrange(q) for _ in range(n)]
            if w not in best:
                best.append(w)
                idx = word_to_idx(w)
                cover[idx] += 1
                if radius >= 1:
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                cover[base + v * step] += 1

    best_len = len(best)
    
    # Helper to update cover when adding/removing a word
    def update_cover_add(cover, w):
        idx = word_to_idx(w)
        cover[idx] += 1
        if radius >= 1:
            for i in range(n):
                cur_sym = w[i]
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        cover[base + v * step] += 1

    def update_cover_remove(cover, w):
        idx = word_to_idx(w)
        cover[idx] -= 1
        if radius >= 1:
            for i in range(n):
                cur_sym = w[i]
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        cover[base + v * step] -= 1

    def try_delete_all(cover, codewords):
        """Aggressively delete any redundant codeword."""
        changed = True
        while changed:
            if time.time() > DEADLINE - 2:
                break
            changed = False
            for ci in range(len(codewords)):
                w = codewords[ci]
                idx = word_to_idx(w)
                if cover[idx] > 1:
                    ok = True
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                if cover[base + v * step] < 2:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        update_cover_remove(cover, w)
                        codewords.pop(ci)
                        changed = True
                        break

    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    alpha = 0.99
    T = T_start
    
    best_code = list(best)
    best_cover = cover.copy()
    
    current_code = list(best)
    current_cover = cover.copy()
    current_len = len(current_code)
    
    # Precompute list of all words? No, too large (6561).
    # But we can generate random words.
    
    while time.time() < DEADLINE - 2:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.4 and len(current_code) > 1:
            # Move: Delete a random codeword
            ci = random.randrange(len(current_code))
            w = current_code[ci]
            update_cover_remove(current_cover, w)
            if is_valid_cover(current_cover):
                # Accept
                current_code.pop(ci)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_code = list(current_code)
                    best_cover = current_cover.copy()
            else:
                # Reject
                update_cover_add(current_cover, w)
        
        elif move_type < 0.7:
            # Move: Insert a random codeword
            # Check if it's already there? 
            # To avoid duplicates, we can track a set of indices.
            # For simplicity, we allow duplicates in the list? No, "codewords must be distinct".
            # So we must check.
            # Using a set of indices for O(1) lookup.
            # We don't maintain a set in the main loop for speed, so we do a linear check?
            # Or we can just try and if it's a duplicate, it's a no-op effectively?
            # No, if we add a duplicate, cover counts go up, but it's not a valid code if we return duplicates.
            # The evaluator checks "codewords must be distinct".
            # So we must ensure distinctness.
            # Let's maintain a set of indices for the current code.
            # This adds overhead.
            # Alternative: Just generate random words and check against a set.
            
            # Let's refactor to use a set of indices for current_code.
            # This might be too slow if we recreate it.
            # Let's just do a linear check for now, n=8, size ~500. 500 checks is fast.
            
            new_word = [random.randrange(q) for _ in range(n)]
            # Check distinctness
            is_dup = False
            for w in current_code:
                if w == new_word:
                    is_dup = True
                    break
            
            if not is_dup:
                update_cover_add(current_cover, new_word)
                current_code.append(new_word)
                new_len = len(current_code)
                
                # Acceptance
                # Delta = new_len - current_len = 1
                # If T is high, we accept.
                # We want to escape local minima, so we accept worse solutions.
                if random.random() < math.exp(-1.0 / T):
                    current_len = new_len
                    # Try to delete immediately to see if we can get back to same or better size
                    try_delete_all(current_cover, current_code)
                    if len(current_code) < best_len:
                        best_len = len(current_code)
                        best_code = list(current_code)
                        best_cover = current_cover.copy()
                else:
                    # Reject
                    current_cover[ word_to_idx(new_word) ] -= 1 # Manual revert? 
                    # Better to use update_cover_remove
                    update_cover_remove(current_cover, new_word)
                    current_code.pop()
                    current_len = len(current_code)
            # If is_dup, do nothing (move failed)
            
        else:
            # Move: Swap a random codeword with a random new word
            if len(current_code) > 0:
                ci = random.randrange(len(current_code))
                old_word = current_code[ci]
                new_word = [random.randrange(q) for _ in range(n)]
                
                # Check if new_word is same as old_word or already in code (excluding old_word)
                # If new_word == old_word, no change.
                # If new_word is in code elsewhere, we'd have duplicates.
                
                # Check distinctness
                is_dup = False
                for j, w in enumerate(current_code):
                    if j != ci and w == new_word:
                        is_dup = True
                        break
                
                if not is_dup and new_word != old_word:
                    # Remove old, add new
                    update_cover_remove(current_cover, old_word)
                    update_cover_add(current_cover, new_word)
                    current_code[ci] = new_word
                    
                    # Size is unchanged.
                    # Accept if valid? It is valid if cover >= 1.
                    # Since we removed old and added new, and old was valid,
                    # new might leave holes if new doesn't cover what old covered and no other code covers those holes.
                    if is_valid_cover(current_cover):
                        # Accept
                        # Try to delete
                        try_delete_all(current_cover, current_code)
                        if len(current_code) < best_len:
                            best_len = len(current_code)
                            best_code = list(current_code)
                            best_cover = current_cover.copy()
                    else:
                        # Reject
                        update_cover_remove(current_cover, new_word)
                        update_cover_add(current_cover, old_word)
                        current_code[ci] = old_word
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Final cleanup
    try_delete_all(best_cover, best_code)
    
    return best_code

# EVOLVE-BLOCK-END
