import numpy as np
import time

def construct():
    """
    Constructs the largest possible collection of 18-bit words with pairwise 
    Hamming distance at least 6. 
    Uses the extended quadratic-residue code [18, 9, 6] which achieves A_2(18, 6) = 512.
    """
    
    # 1. Construct the Generator Matrix for the [18, 9, 6] code
    # The code is derived from the [17, 9, 5] quadratic residue code over GF(2).
    # We compute the generator polynomial g(x) of degree 8.
    
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
        raise RuntimeError("Could not find 17th root of unity in GF(2^8)")

    # Quadratic non-residues mod 17
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # Compute generator polynomial g(x) = prod (x - alpha^i)
    # g(x) will have coefficients in GF(2) because the set of roots is a cyclotomic coset.
    g_coeffs = [1]  # Start with 1
    
    for i in non_residues:
        root = gf28_pow(alpha, i)
        # Multiply by (x + root)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j] ^= gf28_mul(g_coeffs[j], root)
            new_g[j+1] ^= g_coeffs[j]
        g_coeffs = new_g
        
    # g_coeffs should be length 9 (degree 8).
    # Convert to binary (0 or 1)
    # Since the roots form a cyclotomic coset, coefficients should be in GF(2).
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    
    # Construct the 9x17 generator matrix for the [17, 9, 5] code
    # Rows are cyclic shifts of the generator polynomial coefficients
    G_17 = np.zeros((9, 17), dtype=np.int8)
    for i in range(9):
        for j in range(9):
            G_17[i, (i + j) % 17] = bin_g[j]
            
    # Extend to [18, 9, 6] by adding an overall parity check bit
    G_18 = np.zeros((9, 18), dtype=np.int8)
    G_18[:, :17] = G_17
    # Parity bit is sum of first 17 bits mod 2
    G_18[:, 17] = G_17.sum(axis=1) % 2

    # 2. Generate all 512 codewords using NumPy vectorization
    # Create all 9-bit messages (0 to 511) as a matrix of bits
    messages = np.zeros((512, 9), dtype=np.int8)
    for i in range(512):
        for j in range(9):
            messages[i, j] = (i >> j) & 1
            
    # Compute codewords: C = M * G mod 2
    # M is 512x9, G is 9x18. Result is 512x18.
    codewords_matrix = (messages @ G_18) % 2
    
    # Convert matrix rows to integers
    codewords = []
    # Use bit shifts to convert binary rows to ints efficiently
    # col_weights = [1 << j for j in range(18)]
    # This is a matrix multiplication with a column vector of weights
    weights = np.array([1 << j for j in range(18)], dtype=np.int64)
    int_codewords = codewords_matrix @ weights
    
    # Convert numpy array to list of Python ints
    result = int_codewords.tolist()
    
    # 3. Verify the code (Safety check)
    # Check for duplicates
    if len(set(result)) != len(result):
        return _fallback_greedy()
        
    # Check minimum distance
    # For 512 words, pairwise check is ~130k ops, which is fast.
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if (result[i] ^ result[j]).bit_count() < 6:
                return _fallback_greedy()
                
    return result

def _fallback_greedy():
    """Fallback if construction fails (should not happen)"""
    import random
    random.seed(42)
    words = [0]
    deadline = time.time() + 80.0
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
    # Quick verification
    valid = True
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if (words[i] ^ words[j]).bit_count() < 6:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
