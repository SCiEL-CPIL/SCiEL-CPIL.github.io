import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # 1. Construct the [17,8,6] linear code (shortened extended QR code)
    # This code has 2^8 = 256 codewords.

    def gf_mul(a, b, poly=0x1D):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1D):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Construct the generator polynomial for the [17,9,5] QR code
    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]

    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    # 3. Generate all 512 codewords of the [17,9,5] code
    G = np.zeros((9, 17), dtype=np.uint8)

    for k in range(9):
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1

    # Generate codewords using numpy
    words_17 = []
    for u in range(512):
        bits = np.array([(u >> i) & 1 for i in range(9)], dtype=np.uint8)
        word = (bits @ G) % 2
        words_17.append(word)

    # 4. Extend to [18,9,6]
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)

    # 5. Shorten to [17,8,6]
    words_17_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_17_short.append(val)

    if len(words_17_short) != 256:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Precompute popcount LUT for 17-bit integers
    popcount_lut = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount_lut[i] = popcount_lut[i >> 1] + (i & 1)

    # Convert to numpy array for vectorized operations
    code_arr = np.array(words_17_short, dtype=np.uint32)
    n_words = len(code_arr)

    # Precompute all pairwise XORs and their popcounts for the base code
    # This is O(n^2) but n=256 so it's manageable
    # Actually, let's do greedy extension instead

    # Greedy extension: try to add non-linear codewords
    # A candidate is valid if its distance to all existing codewords is >= 6
    # distance(a,b) = popcount(a ^ b) >= 6

    # Try candidates in order of weight (lower weight first tends to work better)
    # Generate candidates by weight
    candidates_by_weight = {}
    for w in range(0, 18):
        # Generate all 17-bit words of weight w
        # Use combinations
        from itertools import combinations
        cands = []
        for positions in combinations(range(17), w):
            word = 0
            for p in positions:
                word |= (1 << p)
            cands.append(word)
        candidates_by_weight[w] = cands

    # Sort candidates by weight, then by value for determinism
    all_candidates = []
    for w in range(0, 18):
        all_candidates.extend(candidates_by_weight[w])

    # Remove any that are already in the code
    code_set = set(words_17_short)
    all_candidates = [c for c in all_candidates if c not in code_set]

    # Greedy addition
    added = 0
    for candidate in all_candidates:
        if time.time() > deadline - 2.0:
            break

        # Check if candidate is at distance >= 6 from all current codewords
        # Use vectorized numpy
        xors = code_arr ^ candidate
        # Compute popcount for each
        # Use the LUT approach via numpy
        # Actually, let's use a faster method
        # popcount via numpy: use a lookup or bit manipulation
        # For 17-bit numbers, we can use:
        # popcount(x) = popcount(x & 0x5555) + popcount((x >> 1) & 0x5555) + ...
        # Or just use the LUT

        # Use numpy to index into LUT
        min_dist = min(popcount_lut[x] for x in xors)

        if min_dist >= 6:
            code_arr = np.append(code_arr, candidate)
            code_set.add(candidate)
            added += 1

    result = code_arr.tolist()
    return result
