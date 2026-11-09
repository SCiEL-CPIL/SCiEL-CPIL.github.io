import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    
    # Total budget in seconds (evaluator kills at 90s, keep margin)
    TOTAL_BUDGET = 60.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # ============================================================
    # Construction: Extended Quadratic Residue Code [18, 9, 6]
    # A_2(18, 6) = 512
    # ============================================================
    # The QR code of length 17 over GF(2) is a [17, 9, 5] code.
    # Generator polynomial: g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # This polynomial corresponds to the cyclotomic coset of 1 modulo 17.
    
    # Represent polynomial as integer. Bit i is coeff of x^i.
    # x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    def poly_mul_mod(m, g, length=17):
        """
        Multiply m(x) * g(x) mod (x^length - 1).
        m is up to 9 bits (deg < 9).
        g is up to 9 bits (deg < 9).
        Result is up to 17 bits.
        """
        result = 0
        # m has at most 9 non-zero terms (bits 0..8)
        for i in range(9):
            if m & (1 << i):
                # Shift g by i
                shifted_g = (g << i)
                # We are working mod x^17 - 1, so if bit >= 17, wrap around
                # deg(m) < 9, deg(g) < 9 => deg(product) < 18
                # Only bit 17 can be set if both high bits are set.
                # x^17 = 1, so x^17 wraps to x^0.
                if shifted_g & (1 << 17):
                    shifted_g ^= (1 << 17)
                    shifted_g |= 1
                # Mask to 17 bits just in case, though logic above handles wrap
                result ^= (shifted_g & 0x1FFFF)
        return result

    qr_codewords_17 = []
    for m in range(512):
        cw = poly_mul_mod(m, g_poly)
        qr_codewords_17.append(cw)

    # Extend to length 18 by appending parity bit
    qr_codewords_18 = []
    for cw in qr_codewords_17:
        parity = cw.bit_count() % 2
        extended = cw | (parity << 17)
        qr_codewords_18.append(extended)

    # Verification
    # Check minimum weight of non-zero codewords
    min_weight = 19
    for cw in qr_codewords_18:
        if cw != 0:
            w = cw.bit_count()
            if w < min_weight:
                min_weight = w

    if min_weight < MIN_DIST:
        # If construction failed (wrong poly?), fallback to greedy
        # But this should not happen with the correct poly.
        pass
    else:
        # Full pairwise check for safety
        # 512 words. 512*511/2 pairs.
        # Using NumPy for speed
        words_arr = np.array(qr_codewords_18, dtype=np.uint32)
        valid = True
        
        # Optimization: Check all pairs
        for i in range(512):
            if time.time() > deadline - 5.0:
                break
            xor_vals = np.bitwise_xor(words_arr, np.uint32(words_arr[i]))
            # Count bits for each
            # Using lookup table or just bit_count?
            # np.vectorize(lambda x: x.bit_count()) is slow.
            # Use popcount trick or just iterate?
            # 512 elements. Python loop is fine.
            for j in range(i + 1, 512):
                d = (words_arr[i] ^ words_arr[j]).bit_count()
                if d < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return qr_codewords_18

    # Fallback: Greedy Search (if QR construction fails for any reason)
    # This is unlikely to beat 512, but ensures a valid result.
    best_words = [0]
    # Simple greedy
    candidates = list(range(1, 2**N))
    candidates.sort(key=lambda x: x.bit_count())
    
    words_list = [0]
    for c in candidates:
        if time.time() > deadline - 5.0:
            break
        ok = True
        for w in words_list:
            if (w ^ c).bit_count() < MIN_DIST:
                ok = False
                break
        if ok:
            words_list.append(c)
            
    if len(words_list) > len(best_words):
        best_words = words_list
        
    return best_words
