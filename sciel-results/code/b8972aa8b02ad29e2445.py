import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generator polynomial for the narrow-sense quadratic residue code of length 17.
    # g(x) = x^8 + x^5 + x^4 + x^2 + 1
    # Non-zero coefficients at indices: 0, 2, 4, 5, 8
    g_indices = [0, 2, 4, 5, 8]
    
    # Construct the 9 basis vectors for the [17, 9, 5] code
    basis = []
    for i in range(9):
        vec = 0
        for j in g_indices:
            vec |= 1 << (i + j)
        basis.append(vec)
        
    # Generate all 512 codewords
    words = []
    for mask in range(512):
        if time.time() > deadline:
            break
        w = 0
        for k in range(9):
            if (mask >> k) & 1:
                w ^= basis[k]
        
        # Extend to length 18 by appending a parity bit
        parity = w.bit_count() % 2
        w_extended = (w << 1) | parity
        words.append(w_extended)
        
    return words
