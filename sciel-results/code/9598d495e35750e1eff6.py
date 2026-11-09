import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Fast popcount for 18-bit values using 3x6-bit lookup
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
    
    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)
    
    # Construct the extended quadratic residue code of length 18
    # QR code for p=17: [17,9,5], extended: [18,9,6] with 512 codewords
    
    # Generator polynomial for QR(17): g(x) = x^8 + x^7 + x^6 + x^5 + x^3 + x^2 + 1
    # Coefficients: g[0]=1, g[2]=1, g[3]=1, g[5]=1, g[6]=1, g[7]=1, g[8]=1
    g_coeffs = [1, 0, 1, 1, 0, 1, 1, 1, 1]  # index = power of x
    
    # Build generator matrix (9x17)
    G = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if g_coeffs[j]:
                pos = (i + j) % 17
                row[pos] = 1
        G.append(row)
    
    # Generate all 2^9 = 512 codewords
    qr_codewords = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= G[i][j]
        # Extend by parity bit
        parity = sum(word) % 2
        word.append(parity)
        
        # Convert to integer
        val = 0
        for j in range(18):
            val |= (word[j] << j)
        qr_codewords.append(val)
    
    best_words = qr_codewords
    best_len = len(best_words)
    
    # Verify minimum distance
    min_weight = 100
    for w in qr_codewords:
        if w != 0:
            wt = w.bit_count()
            if wt < min_weight:
                min_weight = wt
    
    # Try to improve with greedy search
    if time.time() < deadline - 10.0:
        all_candidates = list(range(1, 2**N))
        all_candidates.sort(key=lambda x: (x.bit_count(), x))
        
        rng = np.random.default_rng(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10.0:
                break
            
            candidates = all_candidates[:]
            i = 0
            while i < len(candidates):
                j = i + 1
                while j < len(candidates) and candidates[j].bit_count() == candidates[i].bit_count():
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j
            
            words_list = best_words[:]
            
            for c in candidates:
                if time.time() > deadline - 10.0:
                    break
                
                words_arr = np.array(words_list, dtype=np.uint32)
                xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
                bit_counts = popcount18(xor_vals)
                
                if bit_counts.min() >= MIN_DIST:
                    words_list.append(c)
            
            if len(words_list) > best_len:
                best_len = len(words_list)
                best_words = words_list[:]
            
            if best_len >= 512:
                break
    
    return best_words
