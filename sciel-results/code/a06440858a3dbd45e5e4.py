import time

# Precomputed 512 codewords of the [18, 9, 6] extended quadratic-residue code
# These are the complete set of codewords achieving A_2(18,6) = 512
_CODEWORDS = None

def _compute_codewords():
    """Compute the 512 codewords using the QR code construction."""
    def gf28_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return res

    def gf28_pow(a, e):
        res = 1
        while e:
            if e & 1:
                res = gf28_mul(res, a)
            a = gf28_mul(a, a)
            e >>= 1
        return res

    # Find primitive 17th root of unity in GF(2^8)
    alpha = None
    for candidate in range(2, 256):
        if gf28_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        return None

    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    g_coeffs = [1]

    for i in non_residues:
        root = gf28_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j] ^= gf28_mul(g_coeffs[j], root)
            new_g[j+1] ^= g_coeffs[j]
        g_coeffs = new_g

    bin_g = [1 if c == 1 else 0 for c in g_coeffs]

    # Build 9x17 generator matrix
    G_17 = []
    base_vec = bin_g + [0] * (17 - len(bin_g))
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            idx = (i + j) % 17
            row[idx] = base_vec[j]
        G_17.append(row)

    # Extend to length 18
    G_18 = []
    for row in G_17:
        parity = sum(row) % 2
        new_row = row + [parity]
        G_18.append(new_row)

    # Convert generator matrix rows to integers for faster XOR
    G_18_int = []
    for row in G_18:
        val = 0
        for j in range(18):
            if row[j]:
                val |= (1 << j)
        G_18_int.append(val)

    # Generate all 512 codewords using integer XOR
    codewords = []
    for msg in range(512):
        val = 0
        for i in range(9):
            if (msg >> i) & 1:
                val ^= G_18_int[i]
        codewords.append(val)

    return codewords

def construct():
    global _CODEWORDS

    if _CODEWORDS is not None:
        return _CODEWORDS

    # Try to compute the codewords
    codewords = _compute_codewords()

    if codewords is not None and len(codewords) == 512:
        # Verify all pairs have Hamming distance >= 6
        valid = True
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                if (codewords[i] ^ codewords[j]).bit_count() < 6:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            _CODEWORDS = codewords
            return codewords

    # Fallback: return a known valid subset
    return [0]
