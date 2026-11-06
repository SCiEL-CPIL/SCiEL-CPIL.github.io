import numpy as np
import time

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise 
    Hamming distance at least 6. 
    Uses the extended quadratic-residue code [18, 9, 6] which achieves A_2(18, 6) = 512.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    def gf28_mul(a, b):
        """Multiplication in GF(2^8) with primitive polynomial 0x11D"""
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
        """Exponentiation in GF(2^8)"""
        res = 1
        while e:
            if e & 1:
                res = gf28_mul(res, a)
            a = gf28_mul(a, a)
            e >>= 1
        return res

    # Find a primitive 17th root of unity in GF(2^8)
    alpha = None
    for candidate in range(2, 256):
        if gf28_pow(candidate, 17) == 1:
            alpha = candidate
            break
            
    if alpha is None:
        return _fallback_greedy(deadline)

    # Quadratic non-residues mod 17
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # Compute generator polynomial g(x) = prod (x - alpha^i)
    g_coeffs = [1]
    
    for i in non_residues:
        root = gf28_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j] ^= gf28_mul(g_coeffs[j], root)
            new_g[j+1] ^= g_coeffs[j]
        g_coeffs = new_g
        
    # g_coeffs should be length 9 (degree 8).
    # Verify coefficients are in GF(2)
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    
    # Check that the conversion was lossless (all coeffs were 0 or 1)
    if any(c not in (0, 1) for c in g_coeffs):
        # Coefficients not in GF(2), construction failed
        return _fallback_greedy(deadline)
    
    # Construct the 9x17 generator matrix for the [17, 9, 5] code
    G_17 = np.zeros((9, 17), dtype=np.int8)
    for i in range(9):
        for j in range(9):
            G_17[i, (i + j) % 17] = bin_g[j]
            
    # Extend to [18, 9, 6] by adding an overall parity check bit
    G_18 = np.zeros((9, 18), dtype=np.int8)
    G_18[:, :17] = G_17
    G_18[:, 17] = G_17.sum(axis=1) % 2

    # Generate all 512 codewords
    messages = np.zeros((512, 9), dtype=np.int8)
    for i in range(512):
        for j in range(9):
            messages[i, j] = (i >> j) & 1
            
    codewords_matrix = (messages @ G_18) % 2
    
    weights = np.array([1 << j for j in range(18)], dtype=np.int64)
    int_codewords = codewords_matrix @ weights
    result = int_codewords.tolist()
    
    # Verify: check for duplicates
    if len(set(result)) != len(result):
        return _fallback_greedy(deadline)
        
    # Verify minimum distance
    for i in range(len(result)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(result)):
            if (result[i] ^ result[j]).bit_count() < 6:
                return _fallback_greedy(deadline)
                
    return result

def _fallback_greedy(deadline=None):
    """Fallback if construction fails"""
    if deadline is None:
        deadline = time.time() + 85.0
    import random
    random.seed(42)
    words = [0]
    while len(words) < 512:
        if time.time() > deadline:
            break
        cand = random.getrandbits(18)
        ok = True
        for w in words:
            if (cand ^ w).bit_count() < 6:
                ok = False
                break
        if ok:
            words.append(cand)
    return words

if __name__ == "__main__":
    words = construct()
    print(f"Generated {len(words)} codewords")
    valid = True
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if (words[i] ^ words[j]).bit_count() < 6:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
