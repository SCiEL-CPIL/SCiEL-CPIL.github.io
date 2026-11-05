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

def is_valid_word(candidate, words):
    """Check if candidate has distance >= MIN_DIST from all words in words."""
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    return True

def greedy_add(words, word_set, candidate):
    """Try to add candidate to words if valid. Returns True if added."""
    if candidate in word_set:
        return False
    for w in words:
        if (candidate ^ w).bit_count() < MIN_DIST:
            return False
    words.append(candidate)
    word_set.add(candidate)
    return True

def greedy_random_construction(seed=42, max_iter=1000000, deadline=None, base_words=None):
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
        
        # Generate random word with various weight distributions
        r = rng.random()
        if r < 0.1:
            weight = 0
        elif r < 0.25:
            weight = rng.choice([1, 1, 2, 2, 2])
        elif r < 0.5:
            weight = rng.choice([3, 3, 3, 4, 4, 4, 5])
        elif r < 0.75:
            weight = rng.choice([5, 6, 6, 7, 7, 8])
        elif r < 0.9:
            weight = rng.choice([8, 9, 9, 10, 10])
        else:
            weight = rng.randint(10, N)
        
        positions = rng.sample(range(N), min(weight, N))
        word = 0
        for p in positions:
            word |= 1 << p
        
        if word in word_set:
            continue
        
        if is_valid_word(word, words):
            words.append(word)
            word_set.add(word)
    
    return words

def construct_extended_quadratic_residue_code():
    """
    Constructs the extended quadratic residue code of length 18.
    This is the [18, 9, 6] code, containing 512 codewords.
    """
    # Generator polynomial for QR(17): g(x) = x^8 + x^5 + x^3 + x^2 + 1
    # Coefficients: 1 + x^2 + x^3 + x^5 + x^8
    g_bits = [0] * 17
    g_bits[0] = 1
    g_bits[2] = 1
    g_bits[3] = 1
    g_bits[5] = 1
    g_bits[8] = 1
    
    # Convert to integer for easier manipulation
    g_int = 0
    for i in range(17):
        if g_bits[i]:
            g_int |= (1 << i)
            
    # Generate basis codewords
    # The code is cyclic, so basis vectors are cyclic shifts of g_int
    basis = []
    for i in range(9):
        # Shift g_int by i positions to the right (or left depending on convention)
        # In polynomial representation, multiplying by x^i shifts coefficients.
        # We want the codeword corresponding to message e_i.
        # The codeword is x^i * g(x) mod (x^17 - 1).
        # This is a cyclic shift of the coefficients of g(x).
        
        # Let's just do the shift
        shifted = (g_int >> i) | ((g_int << (17 - i)) & ((1 << 17) - 1)) if i > 0 else g_int
        # Wait, simple shift:
        # If we view g_int as coefficients c_0...c_16 where c_j is bit j.
        # x^i * g(x) has coefficients c_{j-i}.
        # So bit j of the new word is bit (j-i) of g_int.
        # This is a right shift of the integer representation if we align LSB to c_0?
        # Let's check.
        # g_int = sum c_j 2^j.
        # x^i g(x) = sum c_j x^{j+i}.
        # Coefficient of x^k is c_{k-i}.
        # So new_int = sum c_{k-i} 2^k.
        # This is equivalent to (g_int << i) mod (2^17 - 1)?
        # No, (g_int << i) puts c_j at position j+i.
        # We want c_{k-i} at position k.
        # So bit k of result is bit k-i of g_int.
        # This is a right shift of g_int by i?
        # If g_int = 1 (c_0=1), x^1 g(x) = x. Coeff of x^1 is 1.
        # g_int >> 1 = 0. g_int << 1 = 2.
        # So it's a left shift, with wrap-around.
        
        if i == 0:
            shifted = g_int
        else:
            shifted = (g_int << i) | (g_int >> (17 - i))
            shifted &= (1 << 17) - 1
            
        basis.append(shifted)
        
    # Generate all 512 codewords
    words_17 = []
    for msg in range(512):
        cw = 0
        for i in range(9):
            if msg & (1 << i):
                cw ^= basis[i]
        words_17.append(cw)
        
    # Extend to length 18
    words_18 = []
    for cw in words_17:
        weight = cw.bit_count()
        parity = weight % 2
        # Append parity bit at position 17
        extended = cw | (parity << 17)
        words_18.append(extended)
        
    return words_18

def verify_code(words):
    """Verify that all pairwise distances are at least MIN_DIST."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < MIN_DIST:
                return False
    return True

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6.
    
    Strategy:
    1. Directly construct the known optimal [18, 9, 6] extended quadratic residue code (512 words).
    2. Verify the code.
    3. If valid, return it.
    4. If invalid (should not happen), fall back to greedy search.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Try direct construction first
    try:
        words = construct_extended_quadratic_residue_code()
        if verify_code(words):
            return words
        else:
            # Fallback if verification fails
            pass
    except Exception as e:
        print(f"Direct construction failed: {e}")
        
    # Fallback: Greedy search
    best_words = reed_muller_construction()
    
    seeds = [42, 123, 999, 2023]
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=100000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    return best_words

# EVOLVE-BLOCK-END
