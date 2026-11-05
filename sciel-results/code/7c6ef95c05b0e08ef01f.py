import time
import numpy as np

N = 17
MIN_DIST = 6

def _build_linear_code():
    """Build the 256-word [17,8,6] code from the extended QR code shortened by one coordinate.
    
    The [17,9,5] quadratic residue code over GF(2) has generator polynomial
    g(x) = x^8 + x^7 + x^6 + x^4 + x^3 + x + 1.
    Extended by parity to [18,9,6], then shortened by coordinate 0 to get [17,8,6].
    """
    # Generator polynomial coefficients (g[0] = coeff of x^0, ..., g[8] = coeff of x^8)
    # g(x) = x^8 + x^7 + x^6 + x^4 + x^3 + x + 1
    g = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    
    # Generate all 2^9 = 512 codewords
    codewords = []
    for msg in range(512):
        m = np.array([(msg >> i) & 1 for i in range(9)], dtype=np.uint8)
        prod = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if m[i]:
                prod[i:i+9] ^= g
        parity = prod.sum() % 2
        codewords.append(np.append(prod, parity))
    
    # Shorten by coordinate 0: keep codewords with coord 0 = 0, remove coord 0
    shortened = []
    for cw in codewords:
        if cw[0] == 0:
            shortened.append(cw[1:])
    
    # Convert to integers
    words = []
    for cw in shortened:
        w = 0
        for j in range(N):
            if cw[j]:
                w |= (1 << j)
        words.append(w)
    
    # Verify
    assert len(words) == 256, f"Expected 256 words, got {len(words)}"
    assert len(set(words)) == 256, "Duplicate words found"
    
    # Verify minimum distance (spot check)
    words_arr = np.zeros((256, N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            words_arr[i, j] = (w >> j) & 1
    # Check a few pairs
    for _ in range(100):
        i, j = np.random.choice(256, 2, replace=False)
        d = np.bit_count(words_arr[i] ^ words_arr[j])
        assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST} between words {i},{j}"
    
    return words

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    # Build initial 256-word code
    best_words = _build_linear_code()
    best_count = len(best_words)
    
    # Convert to numpy array
    best_arr = np.zeros((best_count, N), dtype=np.uint8)
    for i, w in enumerate(best_words):
        for j in range(N):
            best_arr[i, j] = (w >> j) & 1
    
    word_set = set(best_words)
    rng = np.random.default_rng(12345)
    
    def time_left():
        return deadline - time.time()
    
    # Helper: check if a candidate can be added
    def can_add(candidate_bits):
        """Check if candidate_bits (1D array) has dist >= MIN_DIST from all in best_arr."""
        dists = np.bit_count(candidate_bits[None, :] ^ best_arr)
        return np.min(dists) >= MIN_DIST
    
    # Phase 1: Random search for additional words
    iterations = 0
    while time_left() > 10.0:
        candidate = rng.integers(0, 1 << N)
        if candidate in word_set:
            continue
        
        cand_bits = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        
        if can_add(cand_bits):
            best_words.append(int(candidate))
            word_set.add(int(candidate))
            best_arr = np.vstack([best_arr, cand_bits[None, :]])
            best_count = len(best_words)
        
        iterations += 1
        if iterations % 10000 == 0 and time_left() < 10.0:
            break
    
    # Phase 2: Local search with swaps
    # Try to replace words to enable more additions
    while time_left() > 5.0:
        # Pick a random word to potentially replace
        i = rng.integers(0, len(best_words))
        candidate = rng.integers(0, 1 << N)
        
        if candidate in word_set:
            continue
        
        cand_bits = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        
        # Check distance to all words except i
        if len(best_words) > 1:
            mask = np.ones(len(best_words), dtype=bool)
            mask[i] = False
            dists = np.bit_count(cand_bits[None, :] ^ best_arr[mask])
            if np.min(dists) < MIN_DIST:
                continue
        
        # Replace
        old_word = best_words[i]
        best_words[i] = int(candidate)
        word_set.discard(old_word)
        word_set.add(int(candidate))
        best_arr[i] = cand_bits
        
        # After swap, try to greedily add more words
        for _ in range(200):
            if time_left() < 5.0:
                break
            new_cand = rng.integers(0, 1 << N)
            if new_cand in word_set:
                continue
            nc_bits = np.array([(new_cand >> j) & 1 for j in range(N)], dtype=np.uint8)
            dists = np.bit_count(nc_bits[None, :] ^ best_arr)
            if np.min(dists) >= MIN_DIST:
                best_words.append(int(new_cand))
                word_set.add(int(new_cand))
                best_arr = np.vstack([best_arr, nc_bits[None, :]])
                best_count = len(best_words)
    
    return best_words
