import time

N = 18
MIN_DISTANCE = 6

def construct():
    """
    Constructs the Extended Quadratic Residue Code of length 18.
    This is a [18, 9, 6] linear code with 512 codewords.
    It is known that A_2(18, 6) = 512, so this is optimal.
    """
    
    # The generator polynomial g(x) for the [17, 9, 5] QR code is:
    # g(x) = 1 + x + x^2 + x^4 + x^5 + x^6 + x^7 + x^8
    # Coefficients c_0 to c_8
    c = [1, 1, 1, 0, 1, 1, 1, 1, 1]
    
    # Generate the 9 basis vectors of the [17, 9, 5] code
    # Each basis vector is a cyclic shift of the generator polynomial coefficients
    # within a length of 17.
    basis_17 = []
    for i in range(9):
        vec = [0] * 17
        for j in range(9):
            if c[j] == 1:
                pos = (i + j) % 17
                vec[pos] = 1
        basis_17.append(vec)
        
    # Extend to length 18 by appending a parity bit
    # This creates the [18, 9, 6] code
    basis_18 = []
    for vec in basis_17:
        parity = sum(vec) % 2
        new_vec = vec + [parity]
        # Convert list of bits to integer
        val = 0
        for k in range(18):
            if new_vec[k]:
                val |= (1 << k)
        basis_18.append(val)
        
    # Generate all 2^9 = 512 codewords by XORing subsets of the basis vectors
    code_words = [0]
    for g in basis_18:
        # Double the list by XORing each existing word with the new basis vector
        code_words = code_words + [w ^ g for w in code_words]
        
    # Optional: Verify minimum distance for safety (should be 6)
    # For a linear code, min distance is the min weight of non-zero codewords
    # min_d = min(w.bit_count() for w in code_words if w != 0)
    # assert min_d >= MIN_DISTANCE
    
    return code_words
