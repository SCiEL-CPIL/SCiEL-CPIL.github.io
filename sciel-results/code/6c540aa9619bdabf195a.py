import time

N = 18
MIN_DISTANCE = 6
DEADLINE_SECONDS = 85

def construct():
    """
    Constructs the [18,9,6] Extended Quadratic Residue code.
    
    The [17,8,5] QR code is a cyclic code with generator matrix formed by
    cyclic shifts of the quadratic residue indicator vector mod 17.
    Extending by a parity bit gives [18,8,6]. Adding the all-ones vector
    (which is at distance >= 6 from all codewords in the extended QR code)
    yields the [18,9,6] code with 512 codewords.
    """
    start_time = time.time()
    
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    qrs = [1, 2, 4, 8, 9, 13, 15, 16]
    r0 = 0
    for q in qrs:
        r0 |= (1 << q)
    
    # Build the 8 basis vectors for the [18,8,6] extended QR code
    basis = []
    for i in range(8):
        # Cyclic left shift by i positions (standard circulant convention)
        row = ((r0 << i) | (r0 >> (17 - i))) & ((1 << 17) - 1)
        # Add parity bit at position 17
        parity = row.bit_count() % 2
        extended_row = row | (parity << 17)
        basis.append(extended_row)
    
    # Generate all 256 codewords of the [18,8,6] subcode
    words_256 = []
    for mask in range(256):
        w = 0
        for i in range(8):
            if (mask >> i) & 1:
                w ^= basis[i]
        words_256.append(w)
    
    # Check if all-ones vector can extend the code to [18,9,6]
    all_ones = (1 << 18) - 1
    
    min_d = 19
    for w in words_256:
        d = (all_ones ^ w).bit_count()
        if d < min_d:
            min_d = d
            if min_d < 6:
                break
    
    if min_d >= 6:
        basis.append(all_ones)
    else:
        # Fallback: search for a valid 9th basis vector
        import random
        random.seed(42)
        deadline = start_time + DEADLINE_SECONDS - 5
        for _ in range(10000):
            if time.time() > deadline:
                break
            cand = random.randint(0, (1 << 18) - 1)
            min_d_cand = 19
            for w in words_256:
                d = (cand ^ w).bit_count()
                if d < min_d_cand:
                    min_d_cand = d
                    if min_d_cand < 6:
                        break
            if min_d_cand >= 6:
                basis.append(cand)
                break
    
    # Generate all codewords from the final basis
    dim = len(basis)
    code_words = []
    for mask in range(1 << dim):
        w = 0
        for i in range(dim):
            if (mask >> i) & 1:
                w ^= basis[i]
        code_words.append(w)
    
    return code_words

if __name__ == "__main__":
    words = construct()
    print(f"Code size: {len(words)}")
    min_dist = 19
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
    print(f"Min distance: {min_dist}")
