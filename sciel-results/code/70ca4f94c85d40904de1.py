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
    
    # Precompute binary representation of existing words for fast distance checks
    # words_bin shape: (256, 17)
    words_bin = np.array([[ (w >> i) & 1 for i in range(17) ] for w in words_17_short], dtype=np.uint8)
    n_words = len(words)

    # Helper to check a batch of candidates against the current code
    # Returns list of valid candidate integers
    def check_batch(candidates):
        """
        candidates: numpy array of uint32
        returns: list of candidates that are valid (distance >= 6 from all words)
        """
        if len(candidates) == 0:
            return []
        
        valid = []
        # Convert candidates to binary matrix
        # candidates shape: (N_cand,)
        cand_bin = np.array([[ (c >> i) & 1 for i in range(17) ] for c in candidates], dtype=np.uint8)
        
        # Compute Hamming distances: (N_cand, 17) XOR (N_words, 17) -> need broadcasting
        # cand_bin[:, np.newaxis, :] shape (N_cand, 1, 17)
        # words_bin[np.newaxis, :, :] shape (1, N_words, 17)
        # XOR result shape (N_cand, N_words, 17)
        # Sum over axis 2 -> (N_cand, N_words)
        
        # Memory check: If N_cand is large (e.g., 10000) and N_words is 256, 
        # 10000 * 256 * 17 bytes = ~43 MB. This is acceptable.
        # However, we can do it in chunks if necessary, but let's try full vectorization first.
        
        # Optimization: Only check candidates not already in word_set
        # We'll filter duplicates in the main loop or here.
        
        # To save memory and time, we can process in chunks of candidates
        chunk_size = 5000
        for start in range(0, len(candidates), chunk_size):
            end = min(start + chunk_size, len(candidates))
            chunk_cands = candidates[start:end]
            chunk_bin = cand_bin[start:end]
            
            # Filter out any that are already in word_set (quick check)
            # This is O(N_cand) in Python if done naively, but set lookup is fast.
            # However, we want to keep vectorization. 
            # Let's just compute distances for all, then filter.
            
            xor_matrix = chunk_bin[:, np.newaxis, :] ^ words_bin[np.newaxis, :, :]
            distances = np.sum(xor_matrix, axis=2)
            
            # A candidate is valid if min distance >= 6
            # i.e., all distances >= 6
            # np.all(distances >= 6, axis=1)
            valid_mask = np.all(distances >= 6, axis=1)
            
            for i in range(len(chunk_cands)):
                if valid_mask[i]:
                    c = int(chunk_cands[i])
                    if c not in word_set:
                        valid.append(c)
        
        return valid

    random.seed(12345)
    
    # Strategy 1: Weight-based search
    # Generate all words of weight 6, 7, 8
    # This covers a significant portion of the "boundary" of the code.
    
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
        chunk_size = 2000
        for i in range(0, len(candidates), chunk_size):
            if time.time() > deadline - 5.0:
                break
            chunk = candidates[i:i+chunk_size]
            valids = check_batch(chunk)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                # Update words_bin for future checks? 
                # This is expensive. 
                # Alternative: Keep a list of newly added words and check against them separately?
                # Or rebuild words_bin?
                # Since we are adding very few words (target 258-340, start 256),
                # the growth is small. 
                # However, check_batch uses `words_bin` which is fixed at start.
                # This is a BUG in the current logic: it doesn't check against newly added words.
                # We need to update words_bin or use a different strategy.
                
                # Let's update words_bin dynamically.
                # words_bin is a numpy array. Appending is O(N).
                # Since N is small (< 400), this is fine.
                new_bit = np.array([(v >> i) & 1 for i in range(17)], dtype=np.uint8)
                words_bin = np.vstack([words_bin, new_bit])
                n_words += 1

    # Strategy 2: Random search with vectorized check (if time remains)
    # Use larger batches
    if time.time() < deadline - 5.0:
        batch_size = 2000
        while time.time() < deadline - 2.0:
            candidates = np.array([random.getrandbits(17) for _ in range(batch_size)], dtype=np.uint32)
            
            # Remove duplicates within batch and against existing set
            # check_batch handles the "against existing set" part for the original set,
            # but we updated words_bin. So check_batch will check against ALL current words.
            
            valids = check_batch(candidates)
            for v in valids:
                words = np.append(words, v)
                word_set.add(v)
                new_bit = np.array([(v >> i) & 1 for i in range(17)], dtype=np.uint8)
                words_bin = np.vstack([words_bin, new_bit])
                n_words += 1
                
            # If we found very few, we might be stuck.
            # If we found 0 in a large batch, we might be near the limit.
            
            if len(words) >= 340:
                break

    # Final return as list of integers
    return list(words)
