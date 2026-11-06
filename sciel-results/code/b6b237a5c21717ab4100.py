import time
import numpy as np

def construct():
    """
    Construct the largest possible binary code of length 18 with minimum distance 6.
    Primary strategy: Extended QR code [18, 9, 6] (512 words) + greedy extension.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = 85.0  # Safety margin for 90s limit
    start_time = time.time()
    
    # --- Strategy 1: Extended QR Code [18, 9, 6] ---
    words_18 = _build_qr_code_18()
    
    if words_18 is None:
        return _greedy_construct(N, MIN_DIST, start_time, DEADLINE, seed=None)
    
    # Verify min distance of QR code
    min_dist = 100
    for w in words_18:
        if w != 0:
            wt = w.bit_count()
            if wt < min_dist:
                min_dist = wt
            if min_dist < MIN_DIST:
                break
    
    if min_dist < MIN_DIST:
        return _greedy_construct(N, MIN_DIST, start_time, DEADLINE, seed=None)
    
    # Try to extend with greedy search seeded by QR code
    extended = _greedy_construct(N, MIN_DIST, start_time, DEADLINE, seed=words_18)
    
    if extended is not None and len(extended) > len(words_18):
        return extended
    
    return words_18


def _build_qr_code_18():
    """Build the extended QR(17) code [18, 9, 6] with 512 codewords."""
    # GF(2^8) arithmetic with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D)
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
            b >>= 1
        return res

    def gf_pow(base, exp):
        res = 1
        while exp:
            if exp & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            exp >>= 1
        return res

    # alpha of order 17 in GF(2^8)
    beta = 0x02  # x, primitive element of order 255
    alpha = gf_pow(beta, 15)  # 255/17 = 15, so alpha = beta^15 has order 17

    if gf_pow(alpha, 17) != 1 or alpha == 1:
        return None

    # Cyclotomic coset of 1 mod 17: {1, 2, 4, 8, 16, 15, 13, 9}
    g_coeffs = [1]
    for r in [1, 2, 4, 8, 16, 15, 13, 9]:
        root = gf_pow(alpha, r)
        new_coeffs = [0] * (len(g_coeffs) + 1)
        for i, coef in enumerate(g_coeffs):
            if coef:
                new_coeffs[i + 1] ^= 1
                new_coeffs[i] ^= gf_mul(root, coef)
        g_coeffs = new_coeffs

    # Verify coefficients are in GF(2)
    for c in g_coeffs:
        if c not in (0, 1):
            return None

    # g_coeffs: [g0, g1, ..., g8], degree 8
    g_vec = g_coeffs[:]  # length 9

    # Generate all 512 codewords of length 17
    words_17 = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for k in range(9):
                    pos = i + k
                    if pos < 17 and g_vec[k]:
                        word |= (1 << pos)
        words_17.append(word)

    # Extend to length 18 with parity bit
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        w_ext = (w << 1) | parity
        words_18.append(w_ext)

    return words_18


def _greedy_construct(N, MIN_DIST, start_time, DEADLINE, seed=None):
    """
    Greedy construction using bit-parallel conflict masks.
    If seed is provided, start from those codewords and try to add more.
    """
    size = 1 << N

    # Precompute all masks of weight < MIN_DIST
    masks = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks, dtype=np.uint32)

    forbidden = np.zeros(size, dtype=bool)

    def mark_forbidden(c):
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True

    if seed is not None:
        code = list(seed)
        for c in code:
            mark_forbidden(c)
    else:
        code = [0]
        mark_forbidden(0)

    # Weight-ordered candidate generation for efficiency
    # Group candidates by weight, process low weights first
    # But for 18 bits, we can just iterate in order with time checks
    
    # Try forward order
    for c in range(1, size):
        if time.time() - start_time > DEADLINE:
            break
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)

    # Try reverse order if time permits
    if time.time() - start_time < DEADLINE - 5:
        forbidden[:] = False
        if seed is not None:
            code_rev = list(seed)
            for c in code_rev:
                mark_forbidden(c)
        else:
            code_rev = [0]
            mark_forbidden(0)
        
        for c in range(size - 1, 0, -1):
            if time.time() - start_time > DEADLINE:
                break
            if not forbidden[c]:
                code_rev.append(c)
                mark_forbidden(c)
        
        if len(code_rev) > len(code):
            code = code_rev

    return code
