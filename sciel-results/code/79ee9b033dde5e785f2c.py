import time

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _construct_eqr_18_6():
    """
    Construct the [18, 9, 6] extended quadratic residue code.
    This code has 512 codewords and minimum distance 6.
    It is the known optimal code A_2(18, 6) = 512.
    """
    # Generator matrix for [18, 9, 6] EQR code in systematic form [I | P]
    # Rows are 9-bit vectors for the parity part (columns 9-17)
    # Columns 0-8 are the identity part.
    parity_rows = [
        0b101000110, # Row 0
        0b010100011, # Row 1
        0b110010001, # Row 2
        0b100101000, # Row 3
        0b010010100, # Row 4
        0b001001010, # Row 5
        0b000100101, # Row 6
        0b011101010, # Row 7
        0b101110101, # Row 8
    ]
    
    # Build full 9x18 generator matrix rows as 18-bit integers
    # First 9 bits are identity, last 9 bits are parity
    gen_rows = []
    for i in range(9):
        row = (1 << i) | (parity_rows[i] << 9)
        gen_rows.append(row)
        
    # Enumerate all 2^9 = 512 codewords
    code = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= gen_rows[i]
        code.append(word)
        
    return code

def _greedy_addition(base_words, N=18, min_dist=6, time_limit=5.0, seed=42):
    """
    Fallback greedy addition in case the EQR construction fails verification (unlikely).
    """
    import random
    rng = random.Random(seed)
    best_words = list(base_words)
    start = time.time()
    while time.time() - start < time_limit:
        attempts = 0
        while attempts < 1000 and time.time() - start < time_limit:
            cand = rng.randint(0, (1 << N) - 1)
            ok = True
            for w in best_words:
                if _hamming(cand, w) < min_dist:
                    ok = False
                    break
            if ok:
                best_words.append(cand)
            attempts += 1
    return best_words

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # 1. Deterministic construction of the optimal [18,9,6] EQR code
    code = _construct_eqr_18_6()
    
    # 2. Verify the code meets the requirements
    if len(code) == 512 and _check_min_distance(code, MIN_DIST):
        return code

    # 3. Fallback (should never be reached)
    rm_words = _reed_muller_extended()
    if not _check_min_distance(rm_words, MIN_DIST):
        raise ValueError("RM code failed distance check")

    best = list(rm_words)
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    if time_remaining > 2.0:
        greedy_result = _greedy_addition(best, N, MIN_DIST, time_limit=time_remaining - 1.0, seed=42)
        if len(greedy_result) > len(best) and _check_min_distance(greedy_result, MIN_DIST):
            best = greedy_result

    return best
