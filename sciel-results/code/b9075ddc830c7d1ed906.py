import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Generator polynomial for the narrow-sense Quadratic Residue code of length 17.
    # Roots are alpha^(2^i) for i=0..7, where alpha is a primitive 17th root of unity.
    # The minimal polynomial is g(x) = x^8 + x^5 + x^4 + x^3 + 1.
    # Coefficients (low to high order): [1, 0, 0, 1, 1, 1, 0, 0, 1]
    g_coeffs = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    # Construct the 9 basis vectors for the [17, 9, 5] QR code.
    # Each basis vector is a cyclic shift of the generator vector.
    basis_17 = []
    for shift in range(9):
        row = [0] * 17
        for j, c in enumerate(g_coeffs):
            if c:
                pos = (shift + j) % 17
                row[pos] = 1
        basis_17.append(row)
        
    # Extend to length 18 by adding a parity bit.
    # The extended QR code has parameters [18, 9, 6].
    # Parity bit ensures the sum of bits is 0 (even weight).
    basis_18 = []
    for row in basis_17:
        parity = sum(row) % 2
        new_row = row + [parity]
        # Convert to integer
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        basis_18.append(val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in basis_18:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Verify min distance
    min_d = 19
    for w in code_words:
        if w != 0:
            wt = w.bit_count()
            if wt < min_d:
                min_d = wt
            if min_d < MIN_DISTANCE:
                # Fallback if the hardcoded polynomial was incorrect
                # Return Reed-Muller RM(1,4) which is known to be valid
                rm_basis = []
                b0 = 0
                for i in range(16):
                    b0 |= 1 << i
                rm_basis.append(b0)
                for j in range(4):
                    bj = 0
                    for i in range(16):
                        if (i >> j) & 1:
                            bj |= 1 << i
                    rm_basis.append(bj)
                
                code_words_rm = [0]
                for b in rm_basis:
                    code_words_rm = code_words_rm + [w ^ b for w in code_words_rm]
                return code_words_rm

    return code_words
