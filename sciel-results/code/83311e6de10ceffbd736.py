def construct():
    # Approach: Finite Geometry / Paley Difference Set
    # The [17, 9, 5] quadratic residue code can be generated using the
    # cyclic shifts of the indicator vector of the quadratic residues mod 17.
    
    # Quadratic residues mod 17
    residues = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Base vector of length 17
    # Bit i is 1 if i is in residues
    base = 0
    for r in residues:
        base |= 1 << r
        
    # Generate generator matrix rows (cyclic shifts)
    # The code is cyclic, so the generator matrix consists of 9 cyclic shifts
    # Note: We need to ensure linear independence. For QR codes, the first (p+1)/2 shifts are independent.
    gen_rows = []
    n = 17
    mask = (1 << n) - 1
    
    def cyclic_shift(x, k):
        # Shift left by k, wrapping around n bits
        x = (x << k) | (x >> n)
        return x & mask

    for i in range(9):
        gen_rows.append(cyclic_shift(base, i))
        
    # Generate all 512 codewords
    words_17 = []
    for m in range(512):
        w = 0
        for i in range(9):
            if (m >> i) & 1:
                w ^= gen_rows[i]
        words_17.append(w)
        
    # Extend to length 18 with parity bit
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        w_ext = w | (parity << 17)
        words_18.append(w_ext)
        
    return words_18
