import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (using 55s to be safe)
    
    N = 17
    MIN_DIST = 6
    SIZE = 2 ** N  # 131072
    
    # We'll work with numpy arrays for vectorized operations
    # But for set packing, we need to be careful about memory
    # 131072 x 131072 is too big (17GB), so we can't do full conflict matrix
    
    # Instead, we'll use a greedy approach with efficient distance checking
    
    # First, let's try to construct a good initial code
    # Strategy: Start with the [17,8,6] linear code (256 words) if we can construct it,
    # then try to add more words greedily
    
    # Let's try the QR code construction first
    Q = {1, 4, 5, 6, 7, 9, 11, 16, 17}
    
    # Build the [18,9,6] extended QR code generator
    c = np.zeros(18, dtype=np.uint8)
    for i in range(18):
        if i == 0 or (i % 19) in Q:
            c[i] = 1
    
    # Create circulant matrix and take first 9 rows as generator
    G18 = np.roll(c, np.arange(18)[:, None], axis=1)
    G18_gen = G18[:9]
    
    # Generate all 512 codewords of [18,9,6]
    words_18 = []
    for mask in range(512):
        w = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if mask & (1 << i):
                w ^= G18_gen[i]
        # Convert to integer
        val = 0
        for bit in w:
            val = (val << 1) | bit
        words_18.append(val)
    
    # Check if min distance is at least 6
    min_d = 18
    valid = True
    for i in range(len(words_18)):
        for j in range(i+1, len(words_18)):
            d = (words_18[i] ^ words_18[j]).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        # Shorten to length 17: keep codewords with 0 in last position, remove that bit
        # Actually, let's shorten by removing the first coordinate
        # Shortening: keep codewords where first bit is 0, then remove first bit
        words_17_linear = []
        for w in words_18:
            if (w & 1) == 0:  # Last bit is 0 (LSB)
                words_17_linear.append(w >> 1)
        
        # Verify
        if len(words_17_linear) >= 256:
            min_d_17 = 17
            valid_17 = True
            for i in range(len(words_17_linear)):
                for j in range(i+1, len(words_17_linear)):
                    d = (words_17_linear[i] ^ words_17_linear[j]).bit_count()
                    if d < min_d_17:
                        min_d_17 = d
                        if min_d_17 < MIN_DIST:
                            valid_17 = False
                            break
                if not valid_17:
                    break
            
            if valid_17:
                codewords = set(words_17_linear)
            else:
                codewords = set()
        else:
            codewords = set()
    else:
        codewords = set()
    
    # If linear code construction failed, start with 0
    if len(codewords) == 0:
        codewords = {0}
    
    # Now try to add more words greedily
    # Use a priority-based approach: for each candidate, count how many existing
    # codewords are at distance < MIN_DIST. Add candidates with 0 conflicts.
    
    # To be efficient, we'll iterate through all possible words and check
    # But 131072 candidates x up to 340 codewords is ~45M checks
    # Each check is a XOR + bit_count, which is fast in Python 3.10+
    
    # Let's use a smarter approach: for each existing codeword, mark all words
    # within distance < MIN_DIST as "conflicting". Then find unmarked words.
    
    # But marking all words within distance 5 of a codeword is expensive
    # C(17,0) + C(17,1) + ... + C(17,5) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
    # For 340 codewords, that's ~3.2M markings. Might be feasible.
    
    # Actually, let's just do the greedy approach with early termination
    
    all_words = list(range(SIZE))
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(all_words)
    
    added_count = 0
    last_check = time.time()
    
    for candidate in all_words:
        if time.time() > deadline:
            break
        
        # Check if candidate is already in codewords
        if candidate in codewords:
            continue
        
        # Check distance to all existing codewords
        ok = True
        for cw in codewords:
            if (candidate ^ cw).bit_count() < MIN_DIST:
                ok = False
                break
        
        if ok:
            codewords.add(candidate)
            added_count += 1
        
        # Periodic time check
        if time.time() > deadline:
            break
    
    # Convert to list and return
    result = list(codewords)
    
    # Verify final result
    min_d_final = N
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            if d < min_d_final:
                min_d_final = d
    
    if min_d_final < MIN_DIST:
        # Should not happen, but fallback to just {0}
        return [0]
    
    return result
