import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Step 1: Construct the [17,8,6] linear code from the [18,9,6] QR code
    poly = 0x1D
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
    
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
    
    # Find an element alpha of order 17 in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
    
    if alpha is None:
        return list(range(32))
    
    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) = prod_{i in Q} (x + alpha^i) over GF(2)
    g_coeffs = [1]
    
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
    
    # Build generator matrix G (9 rows, 17 columns)
    G = np.zeros((9, 17), dtype=np.int64)
    
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
    
    # Generate all 512 codewords of the [17,9,5] code
    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)
    
    # Shorten: keep only codewords with 0 in the last (18th) position
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
    
    # Start with the 256-word linear code
    codewords = set(words_short)
    best_count = len(codewords)
    
    if time.time() > deadline - 5:
        return list(codewords)
    
    # Step 2: Try to add words greedily
    # Precompute bit representations
    # For each word, compute its bit vector
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def hamming_dist(a, b):
        return popcount(a ^ b)
    
    # Precompute all words as a list for faster access
    word_list = list(codewords)
    
    # Build a lookup for quick distance checking
    # For each word in the code, find all words at distance 5 or less
    # We'll use a different approach: for each candidate, check against all existing words
    
    # To speed up, we can precompute the "forbidden" set
    # A word is forbidden if it's at distance < 6 from any codeword
    # Let's compute the forbidden set incrementally
    
    # For each existing word, mark all words at distance 0-5 as forbidden
    # But 256 * 9402 = ~2.4M operations, which is feasible
    
    forbidden = set()
    for w in word_list:
        # Enumerate all words at distance 0 to 5 from w
        # Distance 0: just w
        forbidden.add(w)
        # Distance 1: flip each of 17 bits
        for bit in range(17):
            forbidden.add(w ^ (1 << bit))
        # Distance 2: flip pairs
        for b1 in range(17):
            for b2 in range(b1 + 1, 17):
                forbidden.add(w ^ (1 << b1) ^ (1 << b2))
        # Distance 3: flip triples
        for b1 in range(17):
            for b2 in range(b1 + 1, 17):
                for b3 in range(b2 + 1, 17):
                    forbidden.add(w ^ (1 << b1) ^ (1 << b2) ^ (1 << b3))
        # Distance 4 and 5: this gets expensive
        # C(17,4) = 2380, C(17,5) = 6188
        # Total for distance 4-5: ~8568 per word, 256 words = ~2.2M
        # Let's do it but with a time check
        for b1 in range(17):
            for b2 in range(b1 + 1, 17):
                for b3 in range(b2 + 1, 17):
                    for b4 in range(b3 + 1, 17):
                        forbidden.add(w ^ (1 << b1) ^ (1 << b2) ^ (1 << b3) ^ (1 << b4))
        
        if time.time() > deadline - 10:
            break
    
    # Now try to add words not in forbidden
    # Sample random candidates
    rng = np.random.default_rng(42)
    
    max_iter = 500000
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        # Generate a random 17-bit word
        candidate = int(rng.integers(0, 131072))
        if candidate in codewords:
            continue
        if candidate in forbidden:
            continue
        # This word is at distance >= 6 from all existing words
        codewords.add(candidate)
        best_count = len(codewords)
        # Update forbidden set (only need to check if it helps, but it's expensive)
        # Skip forbidden update for speed; the check above is sufficient for now
    
    # Step 3: Try swapping - remove one word and add up to 2 new ones
    # This is complex, so let's try a simpler local search
    # Try to find words that can replace existing words to allow more additions
    
    # Actually, let's try a more targeted approach:
    # For each existing word, try removing it and see if we can add more words
    # This is expensive, so let's do a few random trials
    
    current_list = list(codewords)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try removing a random word
        idx = int(rng.integers(0, len(current_list)))
        removed = current_list[idx]
        trial_set = set(codewords) - {removed}
        
        # Rebuild forbidden for trial_set (expensive, so limit trials)
        # Instead, just try adding words to trial_set
        trial_forbidden = set()
        for w in trial_set:
            trial_forbidden.add(w)
            for bit in range(17):
                trial_forbidden.add(w ^ (1 << bit))
        
        added = 0
        for _ in range(1000):
            candidate = int(rng.integers(0, 131072))
            if candidate in trial_set or candidate in trial_forbidden:
                continue
            # Check distance to all words in trial_set
            ok = True
            for w in trial_set:
                if hamming_dist(candidate, w) < 6:
                    ok = False
                    break
            if ok:
                trial_set.add(candidate)
                added += 1
        
        if len(trial_set) > len(codewords):
            codewords = trial_set
            best_count = len(codewords)
    
    return list(codewords)
