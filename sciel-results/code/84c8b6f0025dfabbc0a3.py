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
    Construct the extended quadratic residue code of length 18.
    This is a [18, 9, 6] code with 512 codewords.
    """
    # The two degree-8 irreducible polynomials over F_2 with roots in the 
    # cyclotomic cosets of 1 and 3 mod 17.
    # One is x^8 + x^5 + x^4 + x^3 + 1
    # The other is x^8 + x^7 + x^6 + x^2 + 1
    candidates = [
        [1, 0, 0, 1, 1, 1, 0, 0, 1], # x^8 + x^5 + x^4 + x^3 + 1
        [1, 0, 1, 0, 0, 0, 1, 1, 1], # x^8 + x^7 + x^6 + x^2 + 1
    ]
    
    for g_coeffs in candidates:
        # Build generator matrix for [17, 9, 5] code
        G_17 = []
        for i in range(9):
            row = 0
            for j in range(9):
                if g_coeffs[j]:
                    pos = (i + j) % 17
                    row |= (1 << pos)
            G_17.append(row)
            
        # Extend to length 18 with parity bit
        G_18 = []
        for row in G_17:
            parity = row.bit_count() % 2
            row_18 = row | (parity << 17)
            G_18.append(row_18)
            
        # Generate all 2^9 = 512 codewords
        words = []
        for mask in range(512):
            word = 0
            for i in range(9):
                if (mask >> i) & 1:
                    word ^= G_18[i]
            words.append(word)
            
        # Verify min distance
        min_d = 18
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                d = hamming_distance(words[i], words[j])
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d < MIN_DIST:
                break
                
        if min_d >= MIN_DIST:
            return words
            
    return None

def greedy_random_construction(seed=42, max_iter=2000000, deadline=None, base_words=None):
    """Greedy random construction to add words to the base code."""
    rng = random.Random(seed)
    if base_words is None:
        words = reed_muller_construction()
    else:
        words = list(base_words)
    word_set = set(words)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Generate random word
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
    Attempts QR code, then tries to extend it with greedy search.
    Falls back to robust greedy search if QR fails.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    best_words = []
    
    # Try QR Code first (fast)
    try:
        qr_words = construct_quadratic_residue_code()
        if qr_words:
            best_words = qr_words
            return best_words
    except Exception:
        pass

    # Fallback: Greedy Search
    seeds = [42, 123, 999, 2023, 777]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=1000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    return best_words

# EVOLVE-BLOCK-END
