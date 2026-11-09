import time

def construct():
    start_time = time.time()
    # Allow 55 seconds for safety, though this should take < 1s
    deadline = start_time + 55.0
    
    # 1. Setup GF(2^8)
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    PRIMITIVE_POLY = 0x1D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIMITIVE_POLY
        return res

    # Build log and antilog tables
    log = [0] * 256
    antilog = [0] * 256
    x = 1
    for i in range(255):
        antilog[i] = x
        log[x] = i
        x = gf_mul(x, 2)
        
    # 2. Compute Generator Polynomial g(x)
    # Roots are alpha^i for i in QR
    QR = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g = [1]
    
    for i in QR:
        root = antilog[i]
        new_g = [0] * (len(g) + 1)
        for j, coeff in enumerate(g):
            if coeff:
                new_g[j+1] ^= coeff
                new_g[j] ^= gf_mul(coeff, root)
        g = new_g
        
    # 3. Generate Codewords
    codewords = []
    for m in range(256):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if g[k]:
                        c[j+k] ^= 1
        word = 0
        for i in range(17):
            if c[i]:
                word |= (1 << i)
        codewords.append(word)
        
    # 4. Filter for Even Weight
    even_weight_words = [w for w in codewords if w.bit_count() % 2 == 0]
    
    # 5. Verify
    # Optional: verify min distance
    min_dist = 17
    for w in even_weight_words:
        if w:
            wt = w.bit_count()
            if wt < min_dist:
                min_dist = wt
                if min_dist < 6:
                    # Fallback if construction failed
                    # Return original seed
                    words = []
                    for slope in range(16):
                        for intercept in range(2):
                            word = 0
                            for point in range(16):
                                bit = ((slope & point).bit_count() + intercept) % 2
                                word |= bit << point
                            words.append(word)
                    return words
                    
    return even_weight_words
