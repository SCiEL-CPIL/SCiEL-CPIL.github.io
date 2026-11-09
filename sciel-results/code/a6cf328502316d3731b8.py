def construct():
    p = 17
    # Quadratic residues mod 17
    qr = set()
    for i in range(1, p):
        qr.add((i*i) % p)
    
    # Generator matrix for [17,9,5] QR code
    base = [0]*17
    base[0] = 1
    for pos in qr:
        base[pos] = 1
    
    G = []
    for i in range(9):
        row = [(base[(pos+i) % p]) for pos in range(p)]
        G.append(row)
    
    # Enumerate all 2^9 codewords
    codewords_17 = []
    for mask in range(2**9):
        word = [0]*17
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    word[j] ^= G[i][j]
        codewords_17.append(word)
    
    # Extend to length 18 by parity bit
    codewords_18 = []
    for word in codewords_17:
        parity = sum(word) % 2
        codewords_18.append(tuple(word + [parity]))
    
    # Shorten by fixing coord 0 to 0
    words_set = set()
    for word in codewords_18:
        if word[0] == 0:
            words_set.add(tuple(word[1:]))
    
    # Convert to integers for faster distance checks
    words_int = set()
    for w in words_set:
        val = 0
        for i, bit in enumerate(w):
            if bit:
                val |= (1 << i)
        words_int.add(val)
    
    # Greedy: try to add more words
    import random
    rng = random.Random(12345)
    
    added = 0
    for _ in range(20000):
        candidate = rng.randint(0, 2**17 - 1)
        if candidate in words_int:
            continue
        
        valid = True
        for w in words_int:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        
        if valid:
            words_int.add(candidate)
            added += 1
            if added >= 100:
                break
    
    return list(words_int)
