import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 18
MIN_DIST = 6

def hamming_distance(a, b):
    return (a ^ b).bit_count()

def reed_muller_construction():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def construct_quadratic_residue_code():
    """
    Construct the [18,9,6] extended quadratic residue code.
    The QR code of length 17 over GF(2) has generator polynomial:
    g(x) = 1 + x^3 + x^4 + x^6 + x^7 + x^9 + x^10 + x^12 + x^16
    Wait, let me use the standard [17,8,5] QR code and extend it.
    
    Actually, the standard approach: The QR code of length 17 (p=17) has
    dimension (p-1)/2 = 8, so [17,8,5]. Extended to [18,8,6].
    But we want 512 = 2^9 words, so we need [18,9,6].
    
    Let me use the known generator matrix for the [18,9,6] code directly.
    The extended QR code of length 18 with 9 information bits.
    """
    # Generator polynomial for the [17,8,5] QR code over GF(2):
    # g(x) = 1 + x^3 + x^4 + x^6 + x^7 + x^9 + x^10 + x^12 + x^16
    # That's degree 9, so it generates a [17, 17-9=8, d] code.
    # Let me verify: the roots are g^1, g^2, ..., g^8 where g is a primitive 17th root.
    # The minimum distance should be 5.
    
    # Actually, let me use a different approach. The [18,9,6] code can be
    # constructed as follows: take the [17,8,5] QR code, extend by one parity
    # bit to get [18,8,6], then add one more systematic bit.
    # No, that doesn't work.
    
    # Let me try: the generator polynomial for the [17,8,5] QR(17) code.
    # g(x) = 1 + x^3 + x^4 + x^6 + x^7 + x^9 + x^10 + x^12 + x^16
    # This has degree 9, so the code has dimension 17-9 = 8.
    # Extended to [18,8,6] by adding an overall parity bit.
    
    # To get 512 = 2^9 words, I need a [18,9,6] code.
    # One way: the [18,9,6] code is the extended QR code with one extra dimension.
    # Actually, A_2(18,6) = 512 is achieved by the extended QR code of length 18.
    # The QR code of length 17 is [17,8,5]. Extended: [18,8,6].
    # But 2^8 = 256, not 512.
    
    # Let me reconsider. Perhaps the [18,9,6] code is constructed differently.
    # It might be the direct sum or a different construction.
    
    # Alternative: Use the known fact that the [18,9,6] code exists and
    # construct it via the generator matrix.
    # The generator matrix for the [18,9,6] extended QR code:
    
    # Let me use the correct generator polynomial. For p=17, the QR code
    # has generator polynomial with roots alpha, alpha^2, ..., alpha^8
    # where alpha is a primitive 17th root of unity in GF(2^8).
    
    # Actually, I'll use a precomputed generator matrix for the [18,9,6] code.
    # The code is the extended QR code. Let me construct it properly.
    
    # The [17,8,5] QR code generator polynomial:
    # g(x) = 1 + x^3 + x^4 + x^6 + x^7 + x^9 + x^10 + x^12 + x^16
    # Coefficients (from x^0 to x^16):
    g_coeffs = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
    # Degree is 16, so the code has dimension 17 - 16 = 1? That's wrong.
    
    # I'm confusing myself. Let me use a completely different approach.
    # I'll use the known generator matrix for the [18,9,6] code.
    
    # The [18,9,6] code is a well-known code. Let me use its generator matrix
    # in systematic form. The first 9 columns form an identity matrix, and
    # the remaining 9 columns form the parity-check part.
    
    # Actually, let me just verify the Reed-Muller code and use a proper
    # greedy search with NumPy vectorization to build up to 512 words.
    
    # For now, return the Reed-Muller code as a base.
    return reed_muller_construction()

def vectorized_greedy_extend(base_words, deadline, seed=42):
    """
    Extend a base code using NumPy-vectorized distance checks.
    Tries candidates in order of increasing weight, then random order.
    """
    rng = random.Random(seed)
    words = list(base_words)
    word_set = set(words)
    
    # Precompute the word array for vectorized operations
    words_arr = np.array(words, dtype=np.uint32)
    
    # Precompute all possible weights
    all_weights = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    # Generate candidates by weight, then shuffle within each weight class
    candidates_by_weight = {}
    for w in all_weights:
        # Generate all C(18, w) candidates
        from itertools import combinations
        cands = []
        for combo in combinations(range(N), w):
            word = 0
            for p in combo:
                word |= 1 << p
            cands.append(word)
        rng.shuffle(cands)
        candidates_by_weight[w] = cands
    
    # Process candidates in order of weight (low weight first for efficiency)
    for w in all_weights:
        if time.time() > deadline:
            break
        
        for candidate in candidates_by_weight[w]:
            if time.time() > deadline:
                break
            
            if candidate in word_set:
                continue
            
            # Vectorized distance check
            if len(words_arr) == 0:
                words.append(candidate)
                word_set.add(candidate)
                words_arr = np.array(words, dtype=np.uint32)
                continue
            
            # Compute XOR distances
            xors = words_arr ^ candidate
            # Count bits using lookup or numpy
            # For uint32, we can use a bit-count trick
            xors = xors.astype(np.uint32)
            # Use a simple approach: convert to list and use bit_count
            # This is still faster than the original due to early termination
            valid = True
            for xor_val in xors:
                if xor_val.bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                words.append(candidate)
                word_set.add(candidate)
                words_arr = np.array(words, dtype=np.uint32)
    
    return words

def greedy_random_construction(seed=42, max_iter=2000000, deadline=None, base_words=None):
    """Greedy random construction to add words to a base code."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Generate random word with bias towards low weights
        weight = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9])
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        valid = True
        for w in words:
            if hamming_distance(word, w) < MIN_DIST:
                valid = False
                break
        if valid:
            words.append(word)
            word_set.add(word)
    
    return words

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    
    Strategy:
    1. Start with Reed-Muller code (32 words, distance 8).
    2. Use vectorized greedy extension to add more words.
    3. Use multiple seeds and deadlines to maximize the count.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Start with Reed-Muller code
    base_words = reed_muller_construction()
    best_words = list(base_words)
    
    # Try vectorized greedy extension with multiple seeds
    seeds = [42, 123, 999, 2023, 777, 555, 111, 888, 333, 666]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        current_words = vectorized_greedy_extend(base_words, deadline, seed=seed)
        if len(current_words) > len(best_words):
            best_words = current_words
    
    # If still small, try pure random greedy with remaining time
    if len(best_words) < 100 and time.time() < deadline:
        for seed in [4242, 1337, 2024]:
            if time.time() > deadline:
                break
            current_words = greedy_random_construction(seed=seed, max_iter=500000, deadline=deadline, base_words=best_words)
            if len(current_words) > len(best_words):
                best_words = current_words
    
    return best_words

# EVOLVE-BLOCK-END
