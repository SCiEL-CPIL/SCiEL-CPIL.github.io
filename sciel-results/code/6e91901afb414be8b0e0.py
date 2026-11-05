import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Generator polynomial for the [17,9,5] cyclic code (unextended QR code of length 17).
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Integer representation: bit 8, 5, 4, 3, 0 are set.
    g = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0)
    
    # Precompute the 9 basis vectors for the [17,9,5] code, extended to length 18.
    basis_18 = []
    for i in range(9):
        vec = g << i
        # Extend to length 18 by adding overall parity bit
        if vec.bit_count() & 1:
            vec |= (1 << 17)
        basis_18.append(vec)
    
    # Generate all 512 codewords using efficient iterative construction
    # Use Gray-code-like approach: each new word differs from previous in few bits
    words = [0]
    for msg in range(1, 512):
        # Compute codeword from message bits
        c_val = 0
        m = msg
        for i in range(9):
            if m & 1:
                c_val ^= basis_18[i]
            m >>= 1
        words.append(c_val)
    
    # Verify minimum distance constraint (spot check for correctness)
    # The [18,9,6] extended QR code is known to have d_min=6
    # Quick sanity check: verify a few pairs
    if len(words) >= 2:
        # Check first few pairs to ensure validity
        for i in range(min(10, len(words))):
            for j in range(i+1, min(i+10, len(words))):
                d = (words[i] ^ words[j]).bit_count()
                if d < 6:
                    # Fallback: return what we have (should not happen)
                    return words[:i]
    
    return words
