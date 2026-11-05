import time
import random

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
            
        # Generate random word
        # Bias towards low weights
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
    1. Try the extended quadratic residue code (512 words) if valid.
    2. If not valid or to improve, use greedy random search with multiple seeds
       and a strict deadline to ensure we return the best result within time.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin from 90s limit
    
    # Try QR Code first (fast, deterministic, potentially high count)
    try:
        # Generator polynomial for the length-17 QR code
        # g(x) = x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
        g_coeffs = [1, 1, 0, 1, 1, 0, 1, 1, 1]
        
        G_17 = []
        for i in range(9):
            row = 0
            for j in range(9):
                bit = g_coeffs[j]
                pos = (i + j) % 17
                if bit:
                    row |= (1 << pos)
            G_17.append(row)
            
        G_18 = []
        for row in G_17:
            parity = row.bit_count() % 2
            row_18 = row | (parity << 17)
            G_18.append(row_18)
            
        qr_words = []
        for mask in range(512):
            word = 0
            for i in range(9):
                if (mask >> i) & 1:
                    word ^= G_18[i]
            qr_words.append(word)
            
        # Verify min distance
        min_d = 18
        valid_qr = True
        for i in range(len(qr_words)):
            for j in range(i+1, len(qr_words)):
                d = hamming_distance(qr_words[i], qr_words[j])
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        valid_qr = False
                        break
            if not valid_qr:
                break
                
        if valid_qr:
            return qr_words
    except Exception:
        pass

    # Fallback: Greedy Search
    # Use multiple seeds to explore different regions of the space
    best_words = []
    seeds = [42, 123, 999, 2023, 777, 555, 111, 888, 333, 666]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        current_words = greedy_random_construction(seed=seed, max_iter=1000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    # If best is still small, try one more long run with a different seed
    if len(best_words) < 100 and time.time() < deadline:
        current_words = greedy_random_construction(seed=4242, max_iter=2000000, deadline=deadline)
        if len(current_words) > len(best_words):
            best_words = current_words
            
    return best_words

# EVOLVE-BLOCK-END
