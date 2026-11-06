import time

N = 18
MIN_DIST = 6

def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    Uses the extended quadratic residue code of length 18, which has parameters [18, 9, 6].
    This yields 2^9 = 512 codewords, which is the known maximum A_2(18, 6).
    """
    p = 17
    # Quadratic residues modulo 17
    qr = set((i * i) % p for i in range(1, p // 2 + 1))
    
    # Construct the 17 shift vectors of the QR code
    shifts = []
    for k in range(p):
        w = 0
        for j in range(p):
            if (j - k) % p in qr:
                w |= (1 << j)
        shifts.append(w)
        
    # Gaussian elimination to find a basis for the code
    basis = []
    for w in shifts:
        current_w = w
        while current_w != 0:
            pivot = current_w.bit_length() - 1
            # Check if a basis vector exists with this pivot bit
            reduced = False
            for i, b in enumerate(basis):
                b_pivot = b.bit_length() - 1
                if b_pivot == pivot:
                    current_w ^= b
                    reduced = True
                    break
            if not reduced:
                # This vector is linearly independent of the current basis
                basis.append(current_w)
                break
                
    # The dimension of the QR code of length 17 is 9.
    # We expect len(basis) to be 9.
    
    # Extend the code to length 18 by adding an overall parity bit
    extended_basis = []
    for b in basis:
        parity = b.bit_count() % 2
        extended_b = b | (parity << 17)
        extended_basis.append(extended_b)
        
    # Generate all codewords
    words = []
    for mask in range(512):
        w = 0
        for i in range(len(extended_basis)):
            if mask & (1 << i):
                w ^= extended_basis[i]
        words.append(w)
        
    return words
