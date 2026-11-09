import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0  # Safety margin (evaluator kills at 90s)
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Step 1: Construct the extended quadratic-residue code [18,9,6] with 512 codewords
    g_val = 313  # binary: 100111001 = 256+32+16+8+1

    def poly_mod(numerator, denominator):
        rem = numerator
        deg_den = denominator.bit_length() - 1
        while rem != 0 and rem.bit_length() - 1 >= deg_den:
            deg_rem = rem.bit_length() - 1
            shift = deg_rem - deg_den
            rem ^= (denominator << shift)
        return rem

    H_val = (1 << 17) - 1
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
                parity = word.bit_count() % 2
                word |= (parity << 16)
                words.append(word)
        return words

    # Construct generator matrix rows (9 rows, 18 bits each)
    rows = []
    for i in range(9):
        vec = g_val << i
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
    best_count = len(base_words)

    # Step 2: Pre-compute all masks with Hamming weight < MIN_DIST (offsets)
    offsets = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            offsets.append(m)
    
    # Step 3: Greedy extension with bitset blocked set
    all_candidates = list(range(1 << N))
    base_set = set(base_words)
    remaining_candidates = [c for c in all_candidates if c not in base_set]
    
    max_restarts = 100 # Can afford more restarts now
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start with the base code
        words = base_words[:]
        words_set = set(words)

        # Build initial blocked bitset
        blocked_bits = 0
        
        # Optimization: Use local variable for speed
        offs = offsets
        for b in base_words:
            # Unroll or use list comprehension? 
            # Loop is fine, but we can try to be clever.
            for o in offs:
                blocked_bits |= (1 << (b ^ o))

        # Random order for remaining candidates
        random.shuffle(remaining_candidates)

        # Greedy extension
        for c in remaining_candidates:
            if time.time() > deadline:
                break

            # Check if blocked
            if (blocked_bits >> c) & 1:
                continue

            # Accept c
            words.append(c)
            words_set.add(c)

            # Block all neighbors
            for o in offs:
                blocked_bits |= (1 << (c ^ o))

        # Update best
        if len(words) > best_count:
            best_words = words[:]
            best_count = len(best_words)

        # If we've reached a very high count, we can stop
        if best_count > 1800:
            break

    return best_words
