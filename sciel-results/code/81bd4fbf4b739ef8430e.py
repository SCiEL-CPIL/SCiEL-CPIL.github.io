import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Pre-compute popcount for all 18-bit values
    popcount_all = np.array([i.bit_count() for i in range(2**N)], dtype=np.uint8)

    def popcount18(arr):
        return popcount_all[arr.astype(np.intp)]

    # ============================================================
    # Construction: Extended Quadratic Residue Code [18,9,6]
    # ============================================================
    # QR(17) code: g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Bit positions: 0, 1, 3, 4, 5, 7, 8
    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    # Fast multiplication mod (x^17 - 1) using precomputed table
    # Precompute all multiples of g_poly
    g_mult_table = np.zeros(512, dtype=np.uint32)
    for m in range(512):
        result = 0
        for i in range(9):
            if m & (1 << i):
                result ^= (g_poly << i)
        # Reduce mod (x^17 - 1): bits 17+ wrap around
        if result & 0x100000:
            result ^= (result & 0x1FFFF)
        g_mult_table[m] = result & 0x1FFFF

    # Extend to length 18 with parity bit
    qr_codewords = np.zeros(512, dtype=np.uint32)
    for i in range(512):
        cw17 = g_mult_table[i]
        parity = cw17.bit_count() % 2
        qr_codewords[i] = cw17 | (parity << 17)

    # Verify the QR construction
    valid = True
    if len(qr_codewords) == 512:
        min_w = min(cw.bit_count() for cw in qr_codewords if cw != 0)
        if min_w < MIN_DIST:
            valid = False
        else:
            # Quick spot check with numpy
            for i in range(0, 512, 64):
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                xor_vals = np.bitwise_xor(qr_codewords, np.uint32(qr_codewords[i]))
                bit_counts = popcount18(xor_vals)
                if bit_counts.min() < MIN_DIST:
                    valid = False
                    break

    if valid:
        # Start with the QR code and try to improve via local search
        best_words = qr_codewords.copy()
        best_len = 512
    else:
        # Fallback: greedy
        best_words = np.array([0], dtype=np.uint32)
        best_len = 1
        candidates = list(range(1, 2**N))
        candidates.sort(key=lambda x: (popcount_all[x], x))
        words_list = [0]
        words_arr = np.array(words_list, dtype=np.uint32)
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
            c_val = np.uint32(c)
            xor_vals = np.bitwise_xor(words_arr, c_val)
            bit_counts = popcount18(xor_vals)
            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)
                words_arr = np.array(words_list, dtype=np.uint32)
        best_words = np.array(words_list, dtype=np.uint32)
        best_len = len(words_list)

    # ============================================================
    # Local Search: Try to add more codewords via swap operations
    # ============================================================
    if best_len >= 512 and time.time() < deadline - 10.0:
        # Build a pool of candidate words not in the code, sorted by weight
        in_code = np.zeros(2**N, dtype=bool)
        in_code[best_words] = True
        pool = []
        for w in range(2**N):
            if not in_code[w] and w.bit_count() >= MIN_DIST:
                pool.append(w)
        pool.sort(key=lambda x: -x.bit_count())  # Higher weight first
        pool = pool[:50000]  # Limit pool size

        if pool and time.time() < deadline - 5.0:
            pool_arr = np.array(pool, dtype=np.uint32)
            current = best_words.copy()
            current_len = best_len
            improved = True
            max_iters = 200
            
            while improved and time.time() < deadline - 5.0 and max_iters > 0:
                improved = False
                max_iters -= 1
                
                # Try to find a codeword to remove and a pool word to add
                # that maintains min distance >= 6
                # For each codeword i, check if removing it allows adding a pool word
                # This is expensive, so sample a subset
                
                # Strategy: try random swaps
                rng = np.random.default_rng(42 + max_iters)
                n_tries = min(100, len(current))
                
                for _ in range(n_tries):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick a random codeword to potentially remove
                    idx_remove = rng.integers(0, len(current))
                    removed_word = current[idx_remove]
                    
                    # Try to find a pool word that can replace it
                    # The new word must be distance >= 6 from all other codewords
                    # (excluding the removed one)
                    other_words = np.delete(current, idx_remove)
                    
                    # Check pool words in batches
                    batch_size = 1000
                    found = False
                    for start in range(0, len(pool_arr), batch_size):
                        if time.time() > deadline - 5.0:
                            break
                        end = min(start + batch_size, len(pool_arr))
                        batch = pool_arr[start:end]
                        
                        # Check distance from batch to other_words
                        # For each batch word, min distance to other_words
                        valid_count = 0
                        for bw in batch[:10]:  # Limit per batch for speed
                            xor_vals = np.bitwise_xor(other_words, np.uint32(bw))
                            bit_counts = popcount18(xor_vals)
                            if bit_counts.min() >= MIN_DIST:
                                # Also check it's not the same as removed_word
                                if bw != removed_word:
                                    # Perform the swap
                                    current[idx_remove] = bw
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                    
                    if found:
                        break

            # After swaps, try to add more words (if we removed any, we might have space)
            # Actually, swaps keep the same size. Let's try pure additions.
            if time.time() < deadline - 5.0:
                # Try to add words one by one
                added = True
                while added and time.time() < deadline - 5.0:
                    added = False
                    for pw in pool:
                        if time.time() > deadline - 5.0:
                            break
                        xor_vals = np.bitwise_xor(current, np.uint32(pw))
                        bit_counts = popcount18(xor_vals)
                        if bit_counts.min() >= MIN_DIST:
                            current = np.append(current, pw)
                            added = True
                            break
                if len(current) > best_len:
                    best_len = len(current)
                    best_words = current.copy()

    return best_words.tolist()
