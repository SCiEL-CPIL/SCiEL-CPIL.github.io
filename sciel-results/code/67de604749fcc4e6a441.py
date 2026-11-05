import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # The Extended Quadratic Residue Code of length 18 is a [18, 9, 6] linear code.
    # It is constructed by extending the [17, 9, 5] QR code with an overall parity bit.
    # The generator polynomial for the [17, 9, 5] QR code is known to be:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # In binary (coeffs for x^8 ... x^0), this is 100111001.
    # Let's represent it as an integer where bit 0 is x^0.
    # g(x) = 1 + 0x + 0x^2 + 1x^3 + 1x^4 + 1x^5 + 0x^6 + 0x^7 + 1x^8
    # Bits: 1 (x^8), 0 (x^7), 0 (x^6), 1 (x^5), 1 (x^4), 1 (x^3), 0 (x^2), 0 (x^1), 1 (x^0)
    # Value: 2^8 + 2^5 + 2^4 + 2^3 + 2^0 = 256 + 32 + 16 + 8 + 1 = 313
    
    g_poly = 313
    length_17 = 17
    
    # Construct the 9 rows of the generator matrix for the [17,9,5] code.
    # Each row is a cyclic shift of the generator polynomial.
    gen_rows_17 = []
    for i in range(9):
        row_val = (g_poly << i) % (1 << length_17)
        gen_rows_17.append(row_val)
        
    # Extend to length 18 by adding an overall parity bit.
    # The parity bit is the XOR sum of the existing bits (mod 2 sum).
    gen_matrix = []
    for row_val in gen_rows_17:
        # Calculate parity of the 17 bits
        parity = row_val.bit_count() % 2
        # Add parity bit as the 18th bit (bit 17)
        extended_val = row_val | (parity << 17)
        gen_matrix.append(extended_val)
        
    # Generate all 512 codewords by taking all linear combinations of the generator rows.
    # Since the code is linear, the set of codewords is the span of the generator rows.
    code_words = [0]
    for g in gen_matrix:
        # Append g XOR each existing codeword
        code_words.extend([w ^ g for w in code_words])
        
    # Verify minimum distance
    # For a linear code, min distance is the minimum weight of non-zero codewords.
    min_d = 19
    for w in code_words:
        if w != 0:
            wt = w.bit_count()
            if wt < min_d:
                min_d = wt
            if min_d < MIN_DISTANCE:
                break
                
    if min_d >= MIN_DISTANCE:
        return code_words
        
    # Fallback: Reed-Muller RM(1,4) which has 32 codewords.
    # This is just a safety net; the QR code should always work.
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
        code_words_rm.extend([w ^ b for w in code_words_rm])
        
    return code_words_rm
