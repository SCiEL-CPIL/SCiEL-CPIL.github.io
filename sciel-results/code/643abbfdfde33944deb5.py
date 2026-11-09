import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = q ** n  # 2187
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Precompute powers of q
    q_pows = [1]
    for i in range(n):
        q_pows.append(q_pows[-1] * q)

    # Precompute ball masks (bitmask of words within Hamming distance 1)
    ball_masks = [0] * total_words
    for i in range(total_words):
        mask = 1 << i
        for pos in range(n):
            current_digit = (i // q_pows[pos]) % q
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                diff = new_digit - current_digit
                new_idx = i + diff * q_pows[pos]
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1

    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        return word

    def word_to_idx(word):
        idx = 0
        for k in range(n):
            idx += word[k] * q_pows[k]
        return idx

    # Fast greedy: maintain uncovered count array
    def greedy_cover(seed_rng):
        uncovered = [1] * total_words  # uncovered count for each word
        code = []
        # For each candidate c, count how many uncovered words it covers
        # We'll track this dynamically
        cover_count = [0] * total_words  # cover_count[c] = # uncovered words in ball(c)
        
        # Initialize cover_count
        for c in range(total_words):
            m = ball_masks[c]
            cnt = 0
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if uncovered[idx] > 0:
                    cnt += 1
                m ^= lsb
            cover_count[c] = cnt

        while True:
            # Find max cover_count
            max_cnt = -1
            best_cands = []
            for c in range(total_words):
                if cover_count[c] > max_cnt:
                    max_cnt = cover_count[c]
                    best_cands = [c]
                elif cover_count[c] == max_cnt:
                    best_cands.append(c)
            
            if not best_cands or max_cnt <= 0:
                break

            chosen = seed_rng.choice(best_cands)
            code.append(chosen)
            
            # Update uncovered and cover_count
            m = ball_masks[chosen]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if uncovered[idx] > 0:
                    uncovered[idx] -= 1
                    if uncovered[idx] == 0:
                        # This word is now covered, decrement cover_count for all c that cover it
                        # Actually we need to decrement cover_count for all candidates whose ball contains idx
                        # This is expensive. Alternative: just recompute cover_count lazily?
                        # For now, we'll just mark it covered and not update cover_count precisely.
                        # Instead, let's use a different approach: just track which words are uncovered
                        pass
                m ^= lsb

            # Simpler: just track uncovered as a set and recompute max each time
            # But that's O(N^2). Let's use the bitmask approach instead.
            break

        # The above is getting complicated. Let's use the simpler bitmask greedy.
        return code

    # Simpler fast greedy using bitmasks
    def greedy_cover_simple(seed_rng):
        uncovered = full_mask
        code = []
        while uncovered:
            max_count = -1
            best_candidates = []
            for c in range(total_words):
                count = (ball_masks[c] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    best_candidates = [c]
                elif count == max_count:
                    best_candidates.append(c)
            
            if not best_candidates or max_count <= 0:
                break
            
            chosen = seed_rng.choice(best_candidates)
            code.append(chosen)
            uncovered &= ~ball_masks[chosen]
        
        return code

    # Local search: prune + 2-opt swaps
    def local_search(code, rng, deadline):
        code = code[:]
        
        # Build coverage count
        count = [0] * total_words
        for c in code:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                count[idx] += 1
                m ^= lsb

        def is_redundant(c):
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if count[idx] < 2:
                    return False
                m ^= lsb
            return True

        def remove_codeword(c):
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                count[idx] -= 1
                m ^= lsb

        def add_codeword(c):
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                count[idx] += 1
                m ^= lsb

        def check_valid_swap_remove(c):
            """Check if removing c keeps all words covered (count >= 1 for all in ball(c))."""
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if count[idx] < 2:
                    return False
                m ^= lsb
            return True

        def check_valid_swap_pair(c1, c2, w1, w2):
            """Check if replacing {c1, c2} with {w1, w2} keeps coverage."""
            # Points in ball(c1) or ball(c2) but not in ball(w1) or ball(w2) must have count >= 2
            # (since we're removing c1 and c2, their coverage drops by 1 each)
            old_mask = ball_masks[c1] | ball_masks[c2]
            new_mask = ball_masks[w1] | ball_masks[w2]
            diff = old_mask & ~new_mask
            m = diff
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if count[idx] < 2:
                    return False
                m ^= lsb
            return True

        def check_valid_swap_2_to_1(c1, c2, w):
            """Check if replacing {c1, c2} with {w} keeps coverage."""
            old_mask = ball_masks[c1] | ball_masks[c2]
            new_mask = ball_masks[w]
            diff = old_mask & ~new_mask
            m = diff
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if count[idx] < 3:  # Need count >= 3 since we remove 2
                    return False
                m ^= lsb
            return True

        # Main local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Prune redundant codewords
            new_code = []
            for c in code:
                if is_redundant(c):
                    remove_codeword(c)
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            
            # Phase 2: Try 2-opt swaps (replace 2 with 2)
            if len(code) >= 2:
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    i1 = rng.randint(0, len(code) - 1)
                    i2 = rng.randint(0, len(code) - 1)
                    if i1 == i2:
                        continue
                    c1 = code[i1]
                    c2 = code[i2]
                    
                    # Pick random new words
                    w1 = rng.randint(0, total_words - 1)
                    w2 = rng.randint(0, total_words - 1)
                    if w1 == w2 or w1 == c1 or w1 == c2 or w2 == c1 or w2 == c2:
                        continue
                    if w1 in code or w2 in code:
                        continue
                    
                    if check_valid_swap_pair(c1, c2, w1, w2):
                        remove_codeword(c1)
                        remove_codeword(c2)
                        add_codeword(w1)
                        add_codeword(w2)
                        code[i1] = w1
                        code[i2] = w2
                        improved = True
                        break

            # Phase 3: Try 2-to-1 swaps (replace 2 with 1)
            if len(code) >= 2:
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    i1 = rng.randint(0, len(code) - 1)
                    i2 = rng.randint(0, len(code) - 1)
                    if i1 == i2:
                        continue
                    c1 = code[i1]
                    c2 = code[i2]
                    
                    w = rng.randint(0, total_words - 1)
                    if w == c1 or w == c2 or w in code:
                        continue
                    
                    if check_valid_swap_2_to_1(c1, c2, w):
                        remove_codeword(c1)
                        remove_codeword(c2)
                        add_codeword(w)
                        # Remove c2 from code (keep w at i1)
                        code[i1] = w
                        code.pop(i2)
                        improved = True
                        break

        return code

    # Main: run multiple restarts
    best_code = None
    best_size = float('inf')
    
    # First, get a good greedy seed
    rng0 = random.Random(0)
    initial_code = greedy_cover_simple(rng0)
    initial_code = local_search(initial_code, random.Random(42), deadline)
    best_code = initial_code[:]
    best_size = len(initial_code)

    # More restarts with different random seeds
    for seed in range(1, 50):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        code = greedy_cover_simple(rng)
        code = local_search(code, rng, deadline)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    
    return result
