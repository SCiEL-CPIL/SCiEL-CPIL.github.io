import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generator polynomial for [17, 9, 5] QR code
    # g(x) = 1 + x + x^3 + x^4 + x^6 + x^7 + x^8
    g_bits = [1, 1, 0, 1, 1, 0, 1, 1, 1]
    
    # Create basis masks (17-bit integers)
    basis_masks = []
    for i in range(9):
        mask = 0
        for j in range(9):
            if g_bits[j]:
                mask |= (1 << ((i + j) % 17))
        basis_masks.append(mask)
        
    words = []
    for m in range(512):
        word_17 = 0
        for i in range(9):
            if (m >> i) & 1:
                word_17 ^= basis_masks[i]
                
        # Parity bit
        parity = word_17.bit_count() % 2
        
        # 18-bit word
        word = word_17
        if parity:
            word |= (1 << 17)
            
        words.append(word)
        
        if time.time() > deadline:
            break
            
    # Verification step (optional but safe)
    # Check min distance for a sample or all pairs?
    # 512 pairs is ~130k checks. Fast enough.
    min_dist = 18
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i+1, n):
            wj = words[j]
            d = (wi ^ wj).bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < 6:
                    # Fallback to Reed-Muller
                    rm_words = []
                    for slope in range(16):
                        for intercept in range(2):
                            word = 0
                            for point in range(16):
                                bit = ((slope & point).bit_count() + intercept) % 2
                                word |= bit << point
                            rm_words.append(word)
                    return rm_words
                    
    return words
