import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Pre-compute popcount for all 18-bit values for fast fallback
    # 2^18 = 262144, fits in memory easily
    popcount_all = np.array([i.bit_count() for i in range(2**N)], dtype=np.uint8)

    def popcount18(arr):
        """
        arr: numpy array of uint32 (or int) values < 2^18
        Returns: numpy array of popcounts
        """
        return popcount_all[arr.astype(np.intp)]

    # ============================================================
    # Construction 1: Extended Quadratic Residue Code [18,9,6]
    # ============================================================
    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    def poly_mul_17(m, g):
        result = 0
        for i in range(9):
            if m & (1 << i):
                result ^= (g << i)
        return result & 0x1FFFF

    qr_codewords_17 = []
    for m in range(512):
        cw = poly_mul_17(m, g_poly)
        qr_codewords_17.append(cw)

    qr_codewords_18 = []
    for cw in qr_codewords_17:
        parity = cw.bit_count() % 2
        extended = cw | (parity << 17)
        qr_codewords_18.append(extended)

    best_words = []
    
    # Verify the QR construction and use as seed
    if len(qr_codewords_18) == 512:
        min_w = min(cw.bit_count() for cw in qr_codewords_18 if cw != 0)
        
        if min_w >= MIN_DIST:
            valid = True
            words_arr = np.array(qr_codewords_18, dtype=np.uint32)
            for i in range(len(words_arr)):
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                xor_vals = np.bitwise_xor(words_arr, np.uint32(words_arr[i]))
                bit_counts = popcount18(xor_vals)
                if bit_counts.min() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                best_words = qr_codewords_18[:]
            else:
                best_words = [] # Fallback if QR is somehow invalid

    # If QR failed or is not the best, we will use greedy search.
    # If QR succeeded, we will try to EXTEND it.
    
    # Prepare candidates
    # If we have a best_words set, we exclude those.
    # If best_words is empty, we include all.
    
    qr_set = set(best_words)
    all_candidates = [c for c in range(2**N) if c not in qr_set]
    
    # Sort candidates by weight to help greedy find dense clusters early?
    # Or just use random shuffles.
    # Let's try a mix: one static weight-sorted pass, then random shuffles.
    
    rng = np.random.default_rng(42)
    
    num_trials = 20
    # We can afford more trials if the QR seed is large, because the initial check is heavier,
    # but adding words reduces the number of candidates that pass the first few checks? No.
    # The cost is dominated by checking candidates against the current set.
    # If current set is 512, cost is ~512 * 260k.
    
    for attempt in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        
        candidates = all_candidates[:]
        
        if attempt == 0 and len(best_words) == 0:
            # If starting from scratch, sort by weight
            candidates.sort(key=lambda x: (popcount_all[x], x))
        else:
            # Random shuffle for diversity
            rng.shuffle(candidates)
        
        current_words = best_words[:]
        current_arr = np.array(current_words, dtype=np.uint32) if current_words else np.array([], dtype=np.uint32)
        
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
            
            c_val = np.uint32(c)
            
            # If current set is empty, we can add anything (but we usually have 0 or QR)
            if len(current_arr) == 0:
                current_words.append(c)
                current_arr = np.array(current_words, dtype=np.uint32)
                continue
            
            xor_vals = np.bitwise_xor(current_arr, c_val)
            bit_counts = popcount18(xor_vals)
            
            if bit_counts.min() >= MIN_DIST:
                current_words.append(c)
                # Update array
                # To avoid O(N) copy every time, we could use a list and convert at the end,
                # but we need the array for the next check.
                # Since N is small (< 1000), np.array() is fast.
                current_arr = np.array(current_words, dtype=np.uint32)
        
        if len(current_words) > len(best_words):
            best_words = current_words[:]

    return best_words
