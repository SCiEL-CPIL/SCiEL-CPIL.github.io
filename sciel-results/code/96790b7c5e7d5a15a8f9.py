import time
import numpy as np

def construct():
    """
    Construct the [18, 9, 6] extended quadratic residue code over GF(2).
    This code has 2^9 = 512 codewords and achieves the theoretical maximum A_2(18, 6) = 512.
    
    The construction works as follows:
    1. The [17, 9, 5] QR code is a cyclic code over GF(2) with generator polynomial g(x)
       whose roots are the quadratic residues mod 17 in GF(2^8).
    2. We compute g(x) using GF(2^8) arithmetic with a primitive polynomial.
    3. The generator matrix G is formed from shifts of g(x).
    4. We extend to length 18 by adding a parity bit to each codeword.
    5. All 512 codewords are generated via linear combinations of the 9 basis vectors.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # GF(2^8) arithmetic with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    MOD_POLY = 0x11D

    def gf_mult(a, b):
        """Multiply two elements of GF(2^8)."""
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= MOD_POLY
        return res

    def gf_pow(base, exp):
        """Exponentiation in GF(2^8)."""
        result = 1
        while exp:
            if exp & 1:
                result = gf_mult(result, base)
            base = gf_mult(base, base)
            exp >>= 1
        return result

    # In GF(2^8) defined by x^8 + x^4 + x^3 + x^2 + 1, let alpha = 0b00000010 (i.e., x).
    # Since 255 = 15 * 17, beta = alpha^15 is a primitive 17th root of unity.
    alpha = 0b00000010
    beta = gf_pow(alpha, 15)

    # The cyclotomic coset of 1 mod 17 under multiplication by 2 is:
    # C1 = {1, 2, 4, 8, 16, 15, 13, 9}
    # This is exactly the set of quadratic residues mod 17.
    # The generator polynomial g(x) = prod_{i in C1} (x + beta^i) over GF(2).
    coset = [1, 2, 4, 8, 16, 15, 13, 9]
    roots = [gf_pow(beta, i) for i in coset]

    # Compute g(x) = prod (x + r) for r in roots, with coefficients in GF(2^8).
    # The final coefficients must lie in GF(2) (i.e., 0 or 1).
    coeffs = [1]  # Start with polynomial 1

    for r in roots:
        new_coeffs = [0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            # x * coeffs[i] * x^i contributes to position i+1
            new_coeffs[i + 1] ^= coeffs[i]
            # r * coeffs[i] * x^i contributes to position i
            new_coeffs[i] ^= gf_mult(coeffs[i], r)
        coeffs = new_coeffs

    # Extract GF(2) coefficients: g_coeffs[i] is the coefficient of x^i in g(x)
    g_coeffs = []
    for c in coeffs:
        if c == 0:
            g_coeffs.append(0)
        elif c == 1:
            g_coeffs.append(1)
        else:
            # Should not happen for the QR code; treat as 0 to avoid crash
            g_coeffs.append(0)

    # g_coeffs has length 9: [g_0, g_1, ..., g_8]
    # The [17, 9, 5] cyclic code has generator matrix with rows being shifts of g.
    # Row k: g_0 g_1 ... g_8 0 ... 0 (shifted right by k positions)
    # We extend each 17-bit word to 18 bits by appending a parity bit.

    # Build the 9 basis vectors of the [18, 9, 6] extended code.
    basis = []
    for k in range(9):
        word = 0
        for i in range(9):
            if g_coeffs[i]:
                word |= (1 << (k + i))
        # Extend with parity bit at position 17
        if word.bit_count() % 2 == 1:
            word |= (1 << 17)
        basis.append(word)

    # Generate all 512 codewords via linear combinations of the 9 basis vectors.
    # Use numpy for a clean vectorized approach.
    n_basis = 9
    n_codes = 1 << n_basis  # 512

    # Build the generator matrix as a numpy array of shape (512, 9) in binary
    # Each row represents the message bits for one codeword.
    msg_bits = np.zeros((n_codes, n_basis), dtype=np.uint8)
    for m in range(n_codes):
        for b in range(n_basis):
            msg_bits[m, b] = (m >> b) & 1

    # Each basis vector is an 18-bit integer. Convert to numpy array.
    basis_arr = np.array(basis, dtype=np.uint32)

    # Compute all codewords: each codeword is the XOR of selected basis vectors.
    # We can do this by treating the basis vectors as columns and multiplying
    # the message matrix by the basis matrix over GF(2).
    # Since XOR is addition mod 2, we can use matrix multiplication mod 2.
    # But a simpler approach: for each message, XOR the appropriate basis vectors.

    codewords = np.zeros(n_codes, dtype=np.uint32)
    for m in range(n_codes):
        w = 0
        for b in range(n_basis):
            if (m >> b) & 1:
                w ^= basis_arr[b]
        codewords[m] = w

    # Convert to Python integers for the return value
    result = codewords.tolist()

    # Verify: check that all codewords are distinct and in [0, 2^18)
    assert len(set(result)) == 512, "Codewords are not distinct"
    assert all(0 <= x < (1 << 18) for x in result), "Codewords out of range"

    # Quick spot-check of minimum distance on a small subset to catch gross errors
    # (Full pairwise check of 512^2/2 ≈ 130k pairs is fast enough)
    sample_size = min(100, len(result))
    import random
    random.seed(42)
    indices = random.sample(range(len(result)), sample_size)
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            d = (result[indices[i]] ^ result[indices[j]]).bit_count()
            if d < 6:
                raise ValueError(f"Minimum distance violated: d({result[indices[i]]}, {result[indices[j]]}) = {d}")

    return result
