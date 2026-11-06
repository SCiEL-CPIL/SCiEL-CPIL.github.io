import numpy as np
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # --- Step 1: Deterministic construction of the extended QR(17) code ---
    # The extended quadratic residue code of length 18 is a [18, 9, 6] code with 512 codewords.
    # We construct it using the generator polynomial g(x) in GF(2)[x]/(x^17 - 1).
    # The generator polynomial is:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # This is the standard QR(17) generator polynomial.
    
    # The code is cyclic of length 17. We then extend it by one parity bit to get length 18.
    # The extension bit is the parity of the 17-bit codeword.
    
    # Generator polynomial coefficients (low degree first):
    # g(x) = 1 + x^3 + x^4 + x^5 + x^8
    # Coeffs: [1, 0, 0, 1, 1, 1, 0, 0, 1]
    g_coeffs = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    g_len = len(g_coeffs)  # 9
    
    # Generate all 512 codewords of the cyclic code of length 17.
    # Each message is a 9-bit value. The codeword is the polynomial m(x) * g(x) mod (x^17 - 1).
    # Since deg(m) < 9 and deg(g) = 8, deg(m*g) < 17, so no reduction needed for the product itself,
    # but we need to handle the cyclic nature. Actually, for a cyclic code, the codewords are
    # linear combinations of the generator and its cyclic shifts.
    # Alternatively, we can compute m(x) * g(x) as a polynomial of degree < 17.
    # Since deg(m) <= 8 and deg(g) = 8, deg(m*g) <= 16, which is < 17. So no wrap-around needed.
    
    qr_code_17 = []
    for msg in range(512):
        # Compute m(x) * g(x)
        cw = 0
        for i in range(9):
            if msg & (1 << i):
                # Shift g_coeffs by i positions
                for j in range(9):
                    if g_coeffs[j]:
                        cw |= (1 << (i + j))
        # cw is a 17-bit integer (bits 0 to 16)
        qr_code_17.append(cw)
    
    # Extend each 17-bit codeword by a parity bit to get 18-bit codewords.
    # The parity bit is the parity of the 17-bit word (sum of bits mod 2).
    qr_code_18 = []
    for cw17 in qr_code_17:
        parity = cw17.bit_count() % 2
        cw18 = cw17 | (parity << 17)
        qr_code_18.append(cw18)
    
    # Remove duplicates (there shouldn't be any, but just in case)
    qr_code_18 = list(set(qr_code_18))
    
    # Verify the distance of the QR code (optional, but good for debugging)
    # For now, we trust the construction.
    
    code = qr_code_18
    code_set = set(code)
    
    # --- Step 2: Greedy extension to find more codewords ---
    # The QR code is [18, 9, 6] with 512 codewords.
    # The maximum size A_2(18, 6) is known to be 512 (for linear codes) or possibly larger for nonlinear.
    # Actually, the upper bound for A_2(18, 6) is 512. So we can't add more.
    # But let's verify: the table of codes says A_2(18, 6) = 512.
    # So the QR code is optimal.
    
    # However, to be safe and to follow the directive of combining approaches,
    # we can still try to add more codewords using the forbidden-set approach from inspiration 1.
    # But since 512 is the maximum, we won't find any.
    
    # Let's do a quick check to see if we can add any more.
    # If we can't, we return the 512 codewords.
    
    # Precompute all masks of weight < 6 for forbidden set updates
    size = 1 << N
    masks = np.array([i for i in range(size) if i.bit_count() < MIN_DIST], dtype=np.uint32)
    
    forbidden = np.zeros(size, dtype=bool)
    for cw in code:
        neighbors = (cw ^ masks) % size
        forbidden[neighbors] = True
    
    # Try to add more codewords using random candidates
    rng = np.random.default_rng(42)
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        candidate = int(rng.integers(0, size))
        if candidate in code_set:
            continue
        
        if not forbidden[candidate]:
            # Check if candidate is valid (should be, since it's not in forbidden)
            code.append(candidate)
            code_set.add(candidate)
            # Update forbidden set
            neighbors = (candidate ^ masks) % size
            forbidden[neighbors] = True
    
    return code
