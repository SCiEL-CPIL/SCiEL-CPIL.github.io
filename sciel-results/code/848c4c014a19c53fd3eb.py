import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 60s limit
    
    # 1. Construct the [17,8,6] linear code (shortened extended QR code) - 256 words
    # Work in GF(2^8)
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
        # Fallback: seed code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) = prod_{i in Q} (x + alpha^i)
    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    # Generator matrix G (9x17) for [17,9,5] QR code
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            bit = g_coeffs[j]
            if bit:
                pos = (j + k) % 17
                G[k, pos] = 1

    # Generate all 512 codewords of [17,9,5]
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    # Extend to [18,9,6] with parity bit
    words_18 = []
    for w in words_17:
        weight = np.sum(w)
        parity = weight % 2
        w_ext = np.append(w, parity)
        words_18.append(w_ext)
        
    # Shorten to [17,8,6]: keep words with 0 at position 17
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
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 2. Vectorized Nonlinear Extension
    words = np.array(words_17_short, dtype=np.uint32)
    word_set = set(words_17_short)
    
    # Precompute bit masks for efficient distance calculation
    # We will use a helper to check distances for a batch of candidates
    
    def check_batch(candidates):
        """
        candidates: numpy array of uint32
        returns: list of candidates that are valid (distance >= 6 from all words)
        """
        if len(candidates) == 0:
            return []
        
        valid = []
        # We can check in chunks to avoid huge memory usage if needed, 
        # but 256 * 10000 is small.
        
        # Convert words to binary matrix for XOR? 
        # Or use bit_count on integers.
        # NumPy doesn't have native popcount for uint32 arrays in all versions, 
        # but we can use a lookup table or bit manipulation.
        # A faster way for small dimensions (17 bits) is to use a precomputed popcount table for 17 bits? 
        # Or just use .bit_count() in a list comprehension if the batch is small?
        # No, vectorization is key.
        
        # Let's use a trick: 
        # For each candidate, we need min distance to all words.
        # We can compute the distance matrix: (N_cand, N_words)
        
        # To speed up popcount in numpy:
        # We can use a lookup table for 8-bit chunks.
        # But simpler: since N_words is only 256, and we want to add *some* words,
        # we can just loop through candidates in Python but use a fast check?
        # No, the prompt asks for improvements.
        
        # Let's try a vectorized popcount using bit shifting and summing.
        # This is slow in Python but fast in C if we can do it in numpy.
        
        # Alternative:
        # Keep the list of words as integers.
        # For a batch of candidates, iterate through the batch.
        # For each candidate, check against all words.
        # This is O(N_cand * N_words).
        # If N_cand is 1000 and N_words is 256, that's 256,000 ops.
        # In Python, this is ~0.1s.
        # In 50s, we can do ~500 batches of 1000. Total 500,000 candidates.
        # This is comparable to the original random search but more structured.
        
        # Let's try to be smarter.
        # We will generate candidates in batches of 5000.
        # We will check them one by one against the set of words, but we will 
        # use a helper that breaks early if distance < 6 is found.
        
        valid_candidates = []
        for c in candidates:
            if c in word_set:
                continue
            is_valid = True
            for w in words:
                if (c ^ w).bit_count() < 6:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(c)
        return valid_candidates

    random.seed(12345)
    
    # Strategy 1: Weight-based search
    # Generate all words of weight 6, 7, 8
    # This covers a significant portion of the "boundary" of the code.
    
    # Generate candidates of specific weights
    # We can use itertools.combinations to generate masks of specific weights
    import itertools
    
    def get_words_of_weight(w):
        masks = []
        for comb in itertools.combinations(range(17), w):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            masks.append(mask)
        return np.array(masks, dtype=np.uint32)
    
    # Try weights 6 through 10
    # Note: The code is symmetric, so we can assume the first bit is 0 or 1?
    # The linear code is not symmetric in that sense (it's linear, so 0 is in it).
    # But we can just check all.
    
    for w in [6, 7, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
            
        candidates = get_words_of_weight(w)
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        # Check in chunks
        chunk_size = 500
        for i in range(0, len(candidates), chunk_size):
            if time.time() > deadline - 5.0:
                break
            chunk = candidates[i:i+chunk_size]
            valids = check_batch(chunk)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                
    # Strategy 2: Random search with vectorized check (if time remains)
    # Use larger batches
    if time.time() < deadline - 5.0:
        batch_size = 2000
        while time.time() < deadline - 2.0:
            candidates = np.array([random.getrandbits(17) for _ in range(batch_size)], dtype=np.uint32)
            # Remove duplicates within batch and against existing set
            # This is expensive to do with numpy, so we do it in the check_batch
            # But check_batch is Python loop.
            
            # Let's optimize check_batch for this case
            # We can filter out candidates already in word_set first
            # But word_set is a Python set.
            
            # Let's just run check_batch
            valids = check_batch(candidates)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                
            # If we found very few, we might be stuck.
            # If we found 0 in a large batch, we might be near the limit.
            # But we know the limit is 258. We started with 256.
            # So we only need 2 more!
            
            if len(words) >= 258:
                break

    # Final return as list of integers
    return list(words)
