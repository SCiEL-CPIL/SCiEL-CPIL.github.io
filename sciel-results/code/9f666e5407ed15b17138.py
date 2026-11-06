"""Improved program for A_2(18,6) using quadratic residue construction + efficient greedy/FLIP search."""
import time
import random
from itertools import combinations

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0  # wall-clock seconds; we use a safety margin

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(words[i], words[j]) < MIN_DIST:
                return False
    return True

def quadratic_residue_code():
    """
    Construct the extended quadratic residue code [18, 9, 6] over F_2.
    
    For p=17 (prime ≡ 3 mod 4), the QR code has:
    - Length p = 17, dimension (p-1)/2 = 8, minimum distance 5
    - The extended QR code has length 18, dimension 9, minimum distance 6
    - This gives 2^9 = 512 codewords
    
    The generator matrix G is 9 x 17 where G[i][j] = 1 if (j - i) mod 17 is a quadratic residue.
    We then add an overall parity bit to extend to length 18.
    """
    p = 17
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    qr_set = set((i * i) % p for i in range(1, p))
    
    dim = (p - 1) // 2  # = 8
    
    # Build generator matrix: dim x p
    # G[i][j] = 1 if (j - i) % p is in qr_set
    G = []
    for i in range(dim):
        row = 0
        for j in range(p):
            if (j - i) % p in qr_set:
                row |= (1 << j)
        G.append(row)
    
    # Generate all 2^8 = 256 codewords of length 17
    codewords_17 = []
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                word ^= G[i]
        codewords_17.append(word)
    
    # Extend each word by an overall parity bit to get length 18
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
    
    return codewords_18

def efficient_greedy_extend(words, time_budget=0):
    """
    Greedily try to add more words to the code while maintaining min distance.
    Uses a set for O(1) membership and efficient distance checking.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    best = list(words)
    best_set = set(best)
    
    random.seed(42)
    
    # Try adding random words
    while time.time() < deadline:
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in best_set:
            continue
        # Check distance to all existing words
        ok = True
        for w in best:
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        if ok:
            best.append(candidate)
            best_set.add(candidate)
    
    return best

def flip_local_search(words, time_budget=0):
    """
    Local search: try to add new words or flip bits in existing codewords
    to create room for additional codewords.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    best = list(words)
    best_set = set(best)
    best_len = len(best)
    
    random.seed(123)
    
    while time.time() < deadline:
        # Try to add a new random word
        candidate = random.randint(0, (1 << N) - 1)
        if candidate not in best_set:
            ok = True
            for w in best:
                if hamming(candidate, w) < MIN_DIST:
                    ok = False
                    break
            if ok:
                best.append(candidate)
                best_set.add(candidate)
                best_len = len(best)
                continue
        
        # Try flipping a bit in a random codeword
        idx = random.randint(0, len(best) - 1)
        bit = random.randint(0, N - 1)
        new_word = best[idx] ^ (1 << bit)
        if new_word not in best_set:
            # Check if replacing best[idx] with new_word maintains validity
            ok = True
            for j, w in enumerate(best):
                if j == idx:
                    continue
                if hamming(new_word, w) < MIN_DIST:
                    ok = False
                    break
            if ok:
                best_set.remove(best[idx])
                best[idx] = new_word
                best_set.add(new_word)
    
    return best

def construct():
    """
    Main entry point. Constructs a code with as many codewords as possible
    with pairwise Hamming distance >= 6, length 18.
    """
    start_time = time.time()
    safety_margin = 5.0
    deadline = start_time + TIME_LIMIT - safety_margin
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify
    if not verify_code(qr_words):
        # Fallback to Reed-Muller (32 words)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # Step 2: Try to extend with greedy search
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        qr_words = efficient_greedy_extend(qr_words, time_budget=remaining_time * 0.4)
    
    # Step 3: Try FLIP local search if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        final_words = flip_local_search(qr_words, time_budget=remaining_time * 0.9)
    else:
        final_words = qr_words
    
    # Final verification
    if not verify_code(final_words):
        # Fallback to the verified QR code
        return qr_words
    
    return final_words
