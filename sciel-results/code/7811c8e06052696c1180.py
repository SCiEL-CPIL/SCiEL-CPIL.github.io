def construct():
    import time
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 85.0

    # The generator polynomial for the [17, 9, 5] cyclic quadratic residue code
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients at bits 8, 5, 4, 3, 0
    g = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    # Verify g(x) divides x^17 + 1
    # x^17 + 1 in binary is (1 << 17) | 1
    target = (1 << 17) | 1
    
    def poly_mod(a, b):
        """Compute a % b in GF(2)[x]"""
        deg_a = a.bit_length() - 1
        deg_b = b.bit_length() - 1
        while deg_a >= deg_b and a != 0:
            shift = deg_a - deg_b
            a ^= (b << shift)
            deg_a = a.bit_length() - 1
        return a

    if poly_mod(target, g) != 0:
        # Fallback to Reed-Muller if hardcoded polynomial is wrong (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Generate basis vectors for the [17, 9, 5] cyclic code
    # Basis: x^i * g(x) for i = 0 to 8
    # Since deg(g) = 8 and i <= 8, deg(x^i * g) <= 16 < 17
    # So no wrap-around needed for the basis vectors themselves
    basis = []
    for i in range(9):
        basis.append(g << i)

    # Generate all 512 codewords
    words = set()
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        
        # Extend to 18 bits: add parity bit at position 17
        # The extended QR code is formed by appending a parity bit to the QR code
        # to make all codewords even weight (since min dist of QR is 5, odd, 
        # extension makes min dist 6)
        parity = word.bit_count() & 1
        word_18 = word | (parity << 17)
        words.add(word_18)
        
    result = list(words)
    
    # Verification
    # Check min distance is at least 6
    # 512 words -> ~130k pairs. Fast.
    min_dist = 100
    for i in range(len(result)):
        wi = result[i]
        for j in range(i + 1, len(result)):
            d = (wi ^ result[j]).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                # Fallback if something went wrong
                words = []
                for slope in range(16):
                    for intercept in range(2):
                        word = 0
                        for point in range(16):
                            bit = ((slope & point).bit_count() + intercept) % 2
                            word |= bit << point
                        words.append(word)
                return words
        if min_dist < 6:
            break

    return result
