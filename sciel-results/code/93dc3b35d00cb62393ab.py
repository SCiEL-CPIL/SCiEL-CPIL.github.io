import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Build the extended quadratic-residue code of length 18, dimension 6, distance 6
    # QR(17): [17, 6, 5] code, extended to [18, 6, 6]
    
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # 0-indexed positions (subtract 1): {0, 1, 3, 7, 8, 12, 14, 15}
    row0_positions = [0, 1, 3, 7, 8, 12, 14, 15]
    
    # Build generator matrix (6 rows, circulant shifts)
    G = []
    for i in range(6):
        row = 0
        for pos in row0_positions:
            shifted = (pos + i) % 17
            row |= (1 << shifted)
        G.append(row)
    
    # Compute all 2^6 = 64 codewords
    qr_words_18 = []
    for mask in range(64):
        word = 0
        for i in range(6):
            if (mask >> i) & 1:
                word ^= G[i]
        # Add parity bit at position 17
        parity = word.bit_count() % 2
        extended = word | (parity << 17)
        qr_words_18.append(extended)
    
    # Verify minimum distance
    def min_distance(words):
        min_d = N + 1
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_d:
                    min_d = d
        return min_d
    
    md = min_distance(qr_words_18)
    if md < MIN_DIST:
        # Fallback to Reed-Muller
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        best_words = list(rm_words)
    else:
        best_words = list(qr_words_18)
    
    best_size = len(best_words)
    
    # Efficient greedy extension with early termination
    def can_add_fast(candidate, current_words):
        for w in current_words:
            if (candidate ^ w).bit_count() < MIN_DIST:
                return False
        return True
    
    current_words = list(best_words)
    current_set = set(current_words)
    
    # Greedy extension: iterate through all 2^18 words
    # Optimize by checking candidates in a smart order
    added_any = True
    while added_any and time.time() < deadline - 8.0:
        added_any = False
        for candidate in range(1 << N):
            if candidate in current_set:
                continue
            if can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                added_any = True
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(best_words)
    
    # Local search: try bit flips on existing codewords
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        flipped_candidates = set()
        for w in current_words:
            for bit in range(N):
                flipped = w ^ (1 << bit)
                if flipped not in current_set:
                    flipped_candidates.add(flipped)
        
        for candidate in flipped_candidates:
            if time.time() > deadline - 3.0:
                break
            if candidate not in current_set and can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(best_words)
    
    # Try 2-bit flips if time permits
    if time.time() < deadline - 4.0:
        two_flip_candidates = set()
        for w in current_words:
            for b1 in range(N):
                for b2 in range(b1 + 1, N):
                    flipped = w ^ (1 << b1) ^ (1 << b2)
                    if flipped not in current_set:
                        two_flip_candidates.add(flipped)
        
        for candidate in two_flip_candidates:
            if time.time() > deadline - 2.0:
                break
            if candidate not in current_set and can_add_fast(candidate, current_words):
                current_words.append(candidate)
                current_set.add(candidate)
                if len(current_words) > best_size:
                    best_words = list(current_words)
                    best_size = len(best_words)
    
    return best_words
