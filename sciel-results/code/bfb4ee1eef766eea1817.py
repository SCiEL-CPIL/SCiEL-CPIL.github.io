import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Use hardcoded generator polynomial coefficients for the Extended QR code of length 18.
    # The QR code of length 17 is a [17, 9, 5] code.
    # Generator polynomial g(x) = 1 + x^3 + x^4 + x^5 + x^8.
    # Coefficients at indices 0, 3, 4, 5, 8 are 1.
    g_coeffs = [0]*9
    g_indices = [0, 3, 4, 5, 8]
    for idx in g_indices:
        g_coeffs[idx] = 1
        
    # Build generator matrix rows
    # The code is cyclic.
    # Generator vector is [g0, g1, ..., g8, 0, ..., 0]
    # Shifts give other rows.
    # Dimension is 17 - 8 = 9.
    
    gen_matrix = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if g_coeffs[j] == 1:
                pos = (i + j) % 17
                row[pos] = 1
        # Extend to length 18 with parity
        parity = sum(row) % 2
        new_row = row + [parity]
        
        val = 0
        for k in range(18):
            if new_row[k]:
                val |= (1 << k)
        gen_matrix.append(val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words = code_words + [w ^ g for w in code_words]
        
    # Verify min distance (linear code: min weight = min distance)
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
        
    # Fallback: Reed-Muller RM(1,4)
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
