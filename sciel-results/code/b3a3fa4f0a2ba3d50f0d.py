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

    # ============================================================
    # Construction 1: Extended Quadratic Residue Code [18,9,6]
    # A_2(18,6) = 512
    # ============================================================
    # The QR code of length 17 over GF(2) is a [17,9,5] code.
    # Generator polynomial: g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Bits at positions: 0,1,3,4,5,7,8

    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    def poly_mul_17(m, g):
        """Multiply m(x)*g(x) mod (x^17 - 1). m is 9-bit, g is 9-bit."""
        result = 0
        for i in range(9):
            if m & (1 << i):
                result ^= (g << i)
        # Reduce mod x^17 - 1 (wrap bits >= 17 around)
        # deg(product) < 17 so no reduction needed, but mask to 17 bits
        return result & 0x1FFFF

    qr_codewords_17 = []
    for m in range(512):
        cw = poly_mul_17(m, g_poly)
        qr_codewords_17.append(cw)

    # Extend to length 18 by appending parity bit
    qr_codewords_18 = []
    for cw in qr_codewords_17:
        parity = cw.bit_count() % 2
        extended = cw | (parity << 17)
        qr_codewords_18.append(extended)

    # Verify: check minimum distance
    min_dist_found = N + 1
    for cw in qr_codewords_18:
        if cw != 0:
            w = cw.bit_count()
            if w < min_dist_found:
                min_dist_found = w

    if min_dist_found >= MIN_DIST:
        # Verify all pairs
        qr_valid = True
        words_arr = np.array(qr_codewords_18, dtype=np.uint32)
        for i in range(len(words_arr)):
            if time.time() > deadline - 10.0:
                break
            xor_vals = np.bitwise_xor(words_arr, np.uint32(words_arr[i]))
            bit_counts = popcount18(xor_vals)
            if bit_counts.min() < MIN_DIST:
                qr_valid = False
                break

        if qr_valid:
            return qr_codewords_18

    # Fallback: greedy search
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))

    best_words = [0]
    best_len = 1

    rng = np.random.default_rng(42)

    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break

        if attempt == 0:
            candidates = all_candidates
        else:
            candidates = all_candidates[:]
            i = 0
            while i < len(candidates):
                j = i + 1
                while j < len(candidates) and candidates[j].bit_count() == candidates[i].bit_count():
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j

        words_list = [0]

        for c in candidates:
            if time.time() > deadline - 5.0:
                break

            words_arr = np.array(words_list, dtype=np.uint32)
            xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
            bit_counts = popcount18(xor_vals)

            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)

        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]

        if best_len >= 500:
            break

    return best_words
