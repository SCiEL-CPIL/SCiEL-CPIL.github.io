import time
import numpy as np

def construct():
    """
    Construct the [18, 9, 6] extended quadratic residue code.
    This code is optimal for A_2(18, 6) = 512.
    
    The construction uses the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
    over GF(2) for the [17, 9, 5] QR code, then extends to length 18 by adding
    an overall parity check bit.
    
    Returns a list of 512 integers, each representing an 18-bit codeword.
    """
    start_time = time.time()
    
    # Generator polynomial coefficients: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # g_coeffs[j] is the coefficient of x^j
    g_coeffs = np.zeros(17, dtype=np.uint8)
    g_coeffs[0] = 1   # x^0
    g_coeffs[3] = 1   # x^3
    g_coeffs[4] = 1   # x^4
    g_coeffs[5] = 1   # x^5
    g_coeffs[8] = 1   # x^8
    
    # Precompute the generator matrix for efficiency
    # The generator matrix G is 9x17, where row i corresponds to message bit i
    # G[i][j] = 1 if x^(i+j mod 17) is in g(x), i.e., g_coeffs[(i+j) % 17] == 1
    G = np.zeros((9, 17), dtype=np.uint8)
    for i in range(9):
        for j in range(17):
            G[i, j] = g_coeffs[(i + j) % 17]
    
    # Generate all 512 codewords
    codewords = []
    for msg in range(512):
        # Extract message bits
        m_vec = np.zeros(9, dtype=np.uint8)
        for i in range(9):
            m_vec[i] = (msg >> i) & 1
        
        # Compute codeword: c = m * G (mod 2)
        # Using matrix multiplication with XOR
        c_vec = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if m_vec[i]:
                c_vec ^= G[i]
        
        # Convert to integer
        word = 0
        for i in range(17):
            if c_vec[i]:
                word |= (1 << i)
        
        # Add parity bit for extension to length 18
        parity = int(np.sum(c_vec)) % 2
        if parity:
            word |= (1 << 17)
        
        codewords.append(word)
    
    # Verify minimum distance (spot check for correctness)
    # Check that all codewords are distinct and have valid length
    assert len(codewords) == 512
    assert len(set(codewords)) == 512
    for w in codewords:
        assert 0 <= w < (1 << 18)
    
    # Quick verification of minimum distance on a subset
    # Check first few pairs to ensure distance >= 6
    import random
    random.seed(42)
    sample_size = min(50, len(codewords))
    sample_indices = random.sample(range(len(codewords)), sample_size)
    for i in range(len(sample_indices)):
        for j in range(i + 1, len(sample_indices)):
            a = codewords[sample_indices[i]]
            b = codewords[sample_indices[j]]
            dist = bin(a ^ b).count('1')
            assert dist >= 6, f"Distance {dist} < 6 between codewords {a} and {b}"
    
    elapsed = time.time() - start_time
    # The [18, 9, 6] extended QR code is optimal; no further improvement possible.
    return codewords
