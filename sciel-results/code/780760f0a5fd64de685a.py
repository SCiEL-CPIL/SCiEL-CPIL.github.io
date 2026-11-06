import time

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0
DEADLINE = time.time() + TIME_LIMIT - 5.0

def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    This is the extended quadratic residue code of length 18, which is a [18, 9, 6] linear code.
    It has 2^9 = 512 codewords.
    """
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Construct the 17 cyclic shifts of the QR indicator vector
    shifts = []
    for k in range(p):
        w = 0
        for j in range(p):
            if (j - k) % p in qr:
                w |= (1 << j)
        shifts.append(w)
    
    # Find a basis of size 9 using span-checking
    basis = []
    for w in shifts:
        if not basis:
            basis.append(w)
            continue
        
        # Check if w is in span of current basis by building all combinations
        span = {0}
        for b in basis:
            new_span = set()
            for s in span:
                new_span.add(s ^ b)
            span = span | new_span
        
        if w not in span:
            basis.append(w)
            if len(basis) == 9:
                break
    
    # Generate all 512 codewords: XOR combinations of basis vectors, then extend with parity bit
    words = []
    for mask in range(512):
        w = 0
        for i in range(9):
            if mask & (1 << i):
                w ^= basis[i]
        
        # Extend to length 18 with overall parity bit
        parity = w.bit_count() % 2
        w18 = w | (parity << 17)
        words.append(w18)
    
    return words
