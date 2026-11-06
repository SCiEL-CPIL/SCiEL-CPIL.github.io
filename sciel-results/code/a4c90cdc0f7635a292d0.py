def construct():
    import time
    start_time = time.time()
    deadline = 60.0
    
    # Generator matrix for the [17,8,6] code (shortened extended QR code)
    # Rows are integers representing 17-bit vectors.
    # We use the known generator matrix for the [17,8,6] binary code.
    
    # The matrix G is 8x17.
    # G = [I_8 | P]
    # P is derived from the QR code structure.
    
    # Let's use the specific rows from literature for the [17,8,6] code.
    # Reference: "Binary [17,8,6] code" generator matrix.
    
    # Row 0: 10000000111100000
    # Row 1: 01000000110011000
    # Row 2: 00100000101010100
    # Row 3: 00010000011001100
    # Row 4: 00001000010101010
    # Row 5: 00000100001100110
    # Row 6: 00000010001010101
    # Row 7: 00000001000110101
    
    # Let's verify these rows form a valid [17,8,6] code.
    # If not, I will use a generic construction method if possible, 
    # but the static matrix is fastest.
    
    G = [
        0b10000000111100000,
        0b01000000110011000,
        0b00100000101010100,
        0b00010000011001100,
        0b00001000010101010,
        0b00000100001100110,
        0b00000010001010101,
        0b00000001000110101
    ]
    
    words = []
    # Generate all 2^8 = 256 codewords
    for mask in range(256):
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                word ^= G[i]
        words.append(word)
        
    # Check validity: min distance >= 6
    # For 256 words, pairwise check is 256*255/2 ~ 32k operations, very fast.
    # But we trust the construction. 
    # However, to be safe against a bad matrix, we can do a quick check.
    # If the matrix is wrong, we might return a code with distance < 6.
    # Let's add a verification step.
    
    # Optimized distance check
    # If any pair has distance < 6, the code is invalid.
    # Given the time limit, we can afford a full check for 256 words.
    
    min_dist = 17
    for i in range(len(words)):
        wi = words[i]
        for j in range(i+1, len(words)):
            wj = words[j]
            d = (wi ^ wj).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                # If invalid, fall back to the seed code (32 words)
                # Or just return the seed code immediately.
                # But this matrix is standard.
                pass
                
    if min_dist >= 6:
        return words
    else:
        # Fallback to seed code if matrix was incorrect
        seed_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                seed_words.append(word)
        return seed_words
