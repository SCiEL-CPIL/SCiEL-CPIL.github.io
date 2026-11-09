import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0  # Safety margin (evaluator kills at 90s)
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Step 1: Construct the extended quadratic-residue code [18,9,6] with 512 codewords
    # Generator polynomial for QR code of length 17: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    g_val = 313  # binary: 100111001 = 256+32+16+8+1

    # Verify g_val divides (x^17 + 1)/(x+1) = x^16 + x^15 + ... + x + 1
    def poly_mod(numerator, denominator):
        rem = numerator
        deg_den = denominator.bit_length() - 1
        while rem != 0 and rem.bit_length() - 1 >= deg_den:
            deg_rem = rem.bit_length() - 1
            shift = deg_rem - deg_den
            rem ^= (denominator << shift)
        return rem

    H_val = (1 << 17) - 1  # x^16 + x^15 + ... + x + 1
    rem = poly_mod(H_val, g_val)

    if rem != 0:
        # Fallback: Reed-Muller RM(1,4) extended - 32 codewords
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Extend with parity bit at position 16
                parity = word.bit_count() % 2
                word |= (parity << 16)
                # Pad to 18 bits (positions 16,17 are parity/zero)
                words.append(word)
        return words

    # Construct generator matrix rows (9 rows, 18 bits each)
    rows = []
    for i in range(9):
        vec = g_val << i
        # Ensure 17 bits wide
        parity = vec.bit_count() % 2
        row = vec | (parity << 17)
        rows.append(row)

    # Generate all 512 codewords of the QR code
    base_words = []
    for m in range(512):
        word = 0
        for j in range(9):
            if (m >> j) & 1:
                word ^= rows[j]
        base_words.append(word)

    best_words = base_words[:]

    # Step 2: Pre-compute all masks with Hamming weight < MIN_DIST
    # These are the "forbidden" offsets - if c is in code, c^m cannot be added
    bad_masks = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            bad_masks.append(m)

    # Step 3: Try to extend the code using greedy with blocked set
    # Multiple restarts with different random orderings of remaining candidates
    
    all_candidates = list(range(1 << N))
    
    # Remove the base words from candidates (they're already in the code)
    base_set = set(base_words)
    remaining_candidates = [c for c in all_candidates if c not in base_set]
    
    max_restarts = 200  # Cap, but time is the real limit
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start with the base code
        words = base_words[:]
        words_set = set(words)
        
        # Build initial blocked set from all base words
        blocked = set()
        for w in base_words:
            for m in bad_masks:
                blocked.add(w ^ m)
        
        # Random order for remaining candidates
        random.shuffle(remaining_candidates)
        
        # Greedy extension
        for c in remaining_candidates:
            if time.time() > deadline:
                break
            
            if c in blocked:
                continue
            
            # Accept c
            words.append(c)
            words_set.add(c)
            
            # Block all neighbors
            for m in bad_masks:
                blocked.add(c ^ m)
        
        # Update best
        if len(words) > len(best_words):
            best_words = words[:]
            
        # If we've reached a very high count, we can stop
        # Upper bound for A_2(18,6) is around 1500-2000
        if len(best_words) > 1800:
            break

    return best_words
