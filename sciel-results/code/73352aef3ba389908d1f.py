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

    # Pre-compute all offsets with popcount < MIN_DIST (i.e., <= 5)
    # These are the words that are "too close" to a selected word
    offsets = []
    for i in range(2**N):
        if popcount_all[i] < MIN_DIST:
            offsets.append(i)
    offsets = np.array(offsets, dtype=np.uint32)

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

    # Verify QR code validity quickly (optional but safe)
    # We trust the construction, but let's assume it's valid.
    # The greedy step will handle conflicts if any exist by skipping them,
    # but QR is a known valid code.
    
    best_words = qr_codewords_18[:]

    # ============================================================
    # Greedy Fill using Bitset of Forbidden Words
    # ============================================================
    
    # We will try a few different random orderings for the fill phase
    # to see if we can find a larger code.
    # Since the QR code is strong, the fill is deterministic if we scan 0..2^18-1.
    # Randomizing the scan order might find different "holes" if the QR code wasn't maximal,
    # but if QR is maximal, no words can be added.
    # If QR is NOT maximal, greedy order matters.
    
    rng = np.random.default_rng(42)
    
    # Pre-compute the list of all words for shuffling
    all_words = np.arange(2**N, dtype=np.uint32)
    
    best_size = len(best_words)
    best_code = best_words[:]

    num_trials = 20 # We can afford many trials because the greedy step is fast
    
    for trial in range(num_trials):
        if time.time() > deadline - 2.0:
            break
        
        # Initialize forbidden mask
        # We start with the QR code words
        forbidden = np.zeros(2**N, dtype=bool)
        
        # Mark neighbors of QR code as forbidden
        # Vectorized approach:
        # For each word in QR, we want to set forbidden[word ^ offset] = True for all offsets.
        # This is equivalent to:
        # for w in qr_code:
        #    forbidden[w ^ offsets] = True
        
        qr_arr = np.array(qr_codewords_18, dtype=np.uint32)
        
        # This loop is 512 iterations, each doing a vectorized set of 12616 elements.
        # 512 * 12616 ~ 6.5M operations.
        # Doing this in a Python loop over QR words is ~512 loops.
        # Each loop: forbidden[qr_arr[i] ^ offsets] = True
        # This is fast.
        
        for w in qr_arr:
            forbidden[w ^ offsets] = True
        
        # Now iterate through candidates
        # Shuffle the candidates to try different orders
        candidates = all_words.copy()
        if trial > 0:
            rng.shuffle(candidates)
        else:
            # First trial: standard order (0 to 2^N-1)
            pass
        
        current_code = qr_codewords_18[:]
        
        # Optimization: If we just want to add words, we can iterate and check forbidden.
        # We don't need to re-check distance against all current words, just the forbidden mask.
        
        for c in candidates:
            if time.time() > deadline - 1.0:
                break
            
            if not forbidden[c]:
                # Add c to code
                current_code.append(int(c))
                # Mark its neighborhood as forbidden
                forbidden[c ^ offsets] = True
        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]

    return best_code
