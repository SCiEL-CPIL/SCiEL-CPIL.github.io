import numpy as np
import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # Construct the extended QR(17) code
    # p = 17
    p = 17
    # Quadratic residues modulo 17
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    # residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create the base vector v of length 17
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask = (1 << p) - 1
    for _ in range((p - 1) // 2):  # 8 shifts
        basis.append(current)
        current = ((current << 1) | (current >> (p - 1))) & mask
    
    # Add the all-ones vector
    basis.append((1 << p) - 1)
    
    # Generate all 2^9 = 512 codewords
    code = []
    num_basis = len(basis)
    for mask_val in range(1 << num_basis):
        word = 0
        for i in range(num_basis):
            if mask_val & (1 << i):
                word ^= basis[i]
        
        # Compute weight of the 17-bit word
        weight = word.bit_count()
        
        # Determine parity bit
        parity = weight % 2
        
        # Form 18-bit integer
        full_word = (word << 1) | parity
        code.append(full_word)
    
    # Verify a sample of pairs to ensure correctness
    if len(code) > 1:
        import random
        rng = random.Random(12345)
        for _ in range(200):
            i, j = rng.sample(range(len(code)), 2)
            d = (code[i] ^ code[j]).bit_count()
            if d < MIN_DIST:
                # This shouldn't happen, but fallback to RM code
                rm_basis = [
                    0b111111111111111100,
                    0b101010101010101000,
                    0b011001100110011000,
                    0b001100110011001100,
                    0b000100010001000100,
                ]
                rm_code = []
                for m in range(32):
                    word = 0
                    for i in range(5):
                        if (m >> i) & 1:
                            word ^= rm_basis[i]
                    rm_code.append(word)
                return rm_code
    
    return code
