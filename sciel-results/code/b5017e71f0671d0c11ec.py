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

def construct_quadratic_residue_code():
    """
    Construct the extended quadratic residue code of length 18.
    This is a [18, 9, 6] code with 512 codewords.
    """
    # The correct generator polynomial for the [17, 9, 5] QR code over GF(2)
    # For p=17, the QR code has generator polynomial g(x) with roots at alpha^i for i in QR(17)
    # QR(17) = {1, 2, 4, 8, 16, 15, 13, 9} (quadratic residues mod 17)
    # The generator polynomial is the product (x - alpha^i) for i in QR(17)
    # Over GF(2), alpha is a primitive 17th root of unity
    
    # The standard [17,9,5] QR code has generator polynomial:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # This gives coefficients (from x^0 to x^8): [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    g_coeffs = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
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
        
    # Verify min distance: for linear code, check min weight of non-zero codewords
    min_weight = 18
    for word in words[1:]:  # Skip zero word
        w = word.bit_count()
        if w < min_weight:
            min_weight = w
            if min_weight < MIN_DIST:
                break
    
    if min_weight >= MIN_DIST:
        return words
        
    return None

def systematic_local_search(words, deadline=None, max_iterations=100000):
    """
    Systematic local search to try to add more words to the code.
    Uses a more structured approach than random greedy.
    """
    if not words:
        return words
    
    words = list(words)
    word_set = set(words)
    n_words = len(words)
    
    # Precompute distances from each existing word to potential new words
    # We'll try words of various weights
    
    for weight in range(0, N + 1):
        if deadline and time.time() > deadline:
            break
        if len(words) >= 512:  # Can't exceed 512 for this problem
            break
            
        # Generate all words of this weight
        # For efficiency, only try if weight is reasonable
        if weight > N:
            continue
            
        # Sample a subset of words of this weight to avoid combinatorial explosion
        if weight <= 3 or weight >= 15:
            # Try all words of this weight (small numbers)
            from itertools import combinations
            positions = list(range(N))
            for combo in combinations(positions, weight):
                if deadline and time.time() > deadline:
                    break
                word = 0
                for p in combo:
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
        else:
            # Sample random words of this weight
            rng = random.Random(weight * 1000 + 42)
            for _ in range(max_iterations // max(1, N - weight)):
                if deadline and time.time() > deadline:
                    break
                if len(words) >= 512:
                    break
                positions = rng.sample(range(N), weight)
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
    Attempts QR code, then tries to extend it with systematic search.
    Falls back to robust greedy search if QR fails.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_words = []
    
    # Try QR Code first (fast)
    try:
        qr_words = construct_quadratic_residue_code()
        if qr_words:
            best_words = qr_words
            # Try to extend the QR code with systematic search
            extended_words = systematic_local_search(qr_words, deadline=deadline, max_iterations=200000)
            if len(extended_words) > len(best_words):
                best_words = extended_words
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
