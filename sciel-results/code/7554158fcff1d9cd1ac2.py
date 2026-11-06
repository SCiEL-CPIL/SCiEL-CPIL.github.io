"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct the extended quadratic residue code of length 18.
    This is a linear [18, 9, 6] code with 512 codewords.
    """
    N = 18
    MIN_DIST = 6
    
    # The generator polynomial for the QR(17) code is g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients for x^0 to x^8: 1, 0, 0, 1, 1, 1, 0, 0, 1
    # As an integer (LSB is x^0): 1 + 8 + 16 + 32 + 256 = 313
    g_poly = 313
    
    # Generate the 9 basis vectors for the length-17 cyclic code
    # The basis vectors are x^i * g(x) for i = 0..8
    # Since deg(g) = 8, deg(x^i * g) = i + 8 <= 16, so no wrap-around needed for i in 0..8
    basis_17 = []
    for i in range(9):
        # Shift g_poly by i bits
        cw = g_poly << i
        # Mask to 17 bits (though not strictly necessary as max bit is 16)
        cw &= (1 << 17) - 1
        basis_17.append(cw)
        
    # Extend to length 18 by adding a parity bit
    # The parity bit is 1 if the weight of the 17-bit vector is odd, 0 otherwise
    # We append it as the 18th bit (bit 17)
    basis_18 = []
    for cw in basis_17:
        weight = cw.bit_count()
        parity = weight % 2
        # Append parity bit at position 17
        cw_18 = cw | (parity << 17)
        basis_18.append(cw_18)
        
    # Generate all 512 codewords by taking all linear combinations of the 9 basis vectors
    code = []
    for mask in range(1 << 9):
        cw = 0
        for i in range(9):
            if (mask >> i) & 1:
                cw ^= basis_18[i]
        code.append(cw)
        
    # The code should have 512 distinct codewords
    # Verify distinctness and minimum distance (optional but good for debugging)
    # For the purpose of the task, we return the code
    return code

# EVOLVE-BLOCK-END
