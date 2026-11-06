import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N
    
    # 1. Precompute popcounts for fast Hamming distance calculation
    # This is crucial for speed. bit_count() is fast, but list lookup is faster in tight loops.
    popcount = [0] * TOTAL_WORDS
    for i in range(1, TOTAL_WORDS):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # 2. Generate the 256-word Quadratic Residue Code seed
    # The [18,9,6] extended QR code shortened by one coordinate gives a [17,8,6] code.
    # Generator polynomial for the [17,8,6] code over GF(2).
    # The QR code of length 17 has generator polynomial g(x) = x^9 + x^7 + x^6 + x^3 + 1.
    # Let's verify this polynomial.
    # Roots are quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16.
    # Actually, standard QR codes of prime length p where p=1 mod 4 have dimension (p-1)/2.
    # For p=17, dim=8.
    # The generator polynomial is the product of (x - alpha^i) for i in Q (quadratic residues).
    # Let's construct it properly or use a known valid generator.
    # A known generator for the [17,8,6] QR code is x^9 + x^7 + x^6 + x^3 + 1.
    # Let's check if this produces 256 distinct codewords with min dist 6.
    
    def generate_qr_code():
        gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
        words = []
        seen = set()
        for msg in range(1 << 8):
            cw = 0
            # Systematic encoding is not strictly necessary for the set, but we need all 256.
            # Simple polynomial multiplication: cw = msg * gen_poly mod (x^17 - 1)? 
            # No, it's a linear code. We just need the span of the generator matrix.
            # The generator matrix rows are shifts of the generator polynomial?
            # For a cyclic code, the codewords are multiples of g(x).
            # Since we are in GF(2), we can just take linear combinations of the basis vectors.
            # The basis vectors are the coefficients of x^i * g(x) mod (x^17-1)?
            # Actually, simpler: just compute all 256 combinations of the 8 basis vectors.
            # Basis vector i corresponds to x^i * g(x) truncated to 17 bits?
            # Let's just use the standard definition: c(x) = m(x)g(x) where deg m < 9.
            # We need to map this to 17-bit integers.
            # g(x) has degree 9. m(x) has degree up to 8. Product has degree up to 17.
            # We take the coefficients of x^0 to x^16.
            
            # To speed up, precompute the 8 basis codewords
            basis = [0] * 8
            for i in range(8):
                # basis[i] is x^i * g(x)
                # g(x) = 1 + x^3 + x^6 + x^7 + x^9
                # x^i * g(x) shifts the bits.
                # We only care about bits 0..16.
                # g(x) bits are at 0, 3, 6, 7, 9.
                # Shifted by i: bits at i, i+3, i+6, i+7, i+9.
                b = 0
                for pos in [0, 3, 6, 7, 9]:
                    bit_pos = i + pos
                    if bit_pos < 17:
                        b |= (1 << bit_pos)
                basis[i] = b
                
            # Now generate all 256 combinations
            for msg in range(256):
                cw = 0
                for i in range(8):
                    if (msg >> i) & 1:
                        cw ^= basis[i]
                if cw not in seen:
                    words.append(cw)
                    seen.add(cw)
        return words

    seed_words = generate_qr_code()
    
    # Verify seed validity
    valid_seed = True
    if len(seed_words) != 256:
        valid_seed = False
    else:
        for i in range(len(seed_words)):
            for j in range(i + 1, len(seed_words)):
                if popcount[seed_words[i] ^ seed_words[j]] < MIN_DIST:
                    valid_seed = False
                    break
            if not valid_seed:
                break

    if valid_seed:
        best_codewords = seed_words
        best_size = 256
    else:
        # Fallback to smaller seed if QR generation failed
        # Use the 32-word RM core from the original program
        best_codewords = []
        seen = set()
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                if word not in seen:
                    best_codewords.append(word)
                    seen.add(word)
        best_size = len(best_codewords)
        # Add 0 if not present? RM(1,4) usually includes 0.
        if 0 not in seen:
            best_codewords.append(0)
            best_size += 1

    # 3. Greedy Extension Function
    def greedy_extend(start_words, order):
        codewords = list(start_words)
        codeword_set = set(codewords)
        
        for candidate in order:
            if time.time() > deadline - 1.0:
                break
            if candidate in codeword_set:
                continue
            
            valid = True
            # Check distance to all existing codewords
            for cw in codewords:
                if popcount[candidate ^ cw] < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                codewords.append(candidate)
                codeword_set.add(candidate)
                
        return codewords

    # 4. Search Loop
    # We will try a natural order first, then random orders.
    
    # Pass 1: Natural order
    natural_order = list(range(TOTAL_WORDS))
    current_words = greedy_extend(best_codewords, natural_order)
    
    if len(current_words) > best_size:
        best_size = len(current_words)
        best_codewords = current_words

    # Pass 2+: Randomized orders
    # We shuffle the full range and try to add words not already in best_codewords.
    # To save time, we can just shuffle the full list and skip existing ones.
    
    all_candidates = list(range(TOTAL_WORDS))
    
    while time.time() < deadline - 2.0:
        random.shuffle(all_candidates)
        current_words = greedy_extend(best_codewords, all_candidates)
        
        if len(current_words) > best_size:
            best_size = len(current_words)
            best_codewords = current_words
            
            # If we hit the upper bound, we can stop early
            if best_size >= 340:
                break

    return best_codewords
