import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    RADIUS = MIN_DIST - 1  # 5
    
    # Precompute all vectors in GF(2)^N with weight <= RADIUS
    ball_offsets = []
    for i in range(2**N):
        if i.bit_count() <= RADIUS:
            ball_offsets.append(i)
    
    ball_offsets = np.array(ball_offsets, dtype=np.uint32)
    
    # Step 1: Generate the [17,8,5] QR code
    # Quadratic residues mod 17
    qr = [1, 2, 4, 8, 9, 13, 15, 16]
    b = np.zeros(17, dtype=int)
    for q in qr:
        b[q] = 1
        
    # Generate basis vectors (cyclic shifts)
    basis = []
    for i in range(8):
        row = np.roll(b, i)
        basis.append(row)
        
    G = np.array(basis)
    
    # Generate all 256 codewords
    codewords_linear = []
    for mask in range(256):
        # Compute linear combination
        c = np.zeros(17, dtype=int)
        for i in range(8):
            if (mask >> i) & 1:
                c = (c + G[i]) % 2
        
        # Convert to integer
        val = 0
        for bit in c:
            val = (val << 1) | int(bit)
        codewords_linear.append(val)
        
    # Check min distance of linear code
    min_d = 100
    for i in range(len(codewords_linear)):
        for j in range(i+1, len(codewords_linear)):
            d = (codewords_linear[i] ^ codewords_linear[j]).bit_count()
            if d < min_d:
                min_d = d
                if min_d < 5:
                    break
        if min_d < 5:
            break
            
    # If min distance is 5, we use the even-weight subcode to get distance 6
    # The even-weight subcode consists of codewords with even parity.
    # Since the code is linear, this is a subcode of dimension 8 (if there are odd weight words).
    # If min_d == 5, the odd weight words have weight 5.
    # The even weight words have weight >= 6 (since 5 is odd).
    # So the even subcode has min distance >= 6.
    
    if min_d == 5:
        # Filter for even weight
        codewords = [c for c in codewords_linear if c.bit_count() % 2 == 0]
    elif min_d >= 6:
        codewords = codewords_linear
    else:
        # Fallback to greedy if construction failed
        codewords = [0]
        forbidden = np.zeros(2**N, dtype=bool)
        forbidden[ball_offsets] = True
        while time.time() < deadline:
            available = np.where(~forbidden)[0]
            if len(available) == 0:
                break
            new_word = int(available[0])
            codewords.append(new_word)
            new_forbidden = new_word ^ ball_offsets
            forbidden[new_forbidden] = True
        return codewords
        
    # Step 2: Try to add non-linear words to the 256-word code
    forbidden = np.zeros(2**N, dtype=bool)
    for c in codewords:
        forbidden[c ^ ball_offsets] = True
        
    # Find available candidates
    available = np.where(~forbidden)[0]
    
    # Greedily add as many as possible
    while len(available) > 0 and time.time() < deadline:
        # Pick a random candidate or the first one?
        # Random might help find a valid addition if the first one is "bad" in some sense?
        # Actually, any available word is valid to add.
        # Adding one might block others.
        # Greedy: pick the one that blocks the fewest?
        # For now, pick the first one (lowest value) to be deterministic.
        # Or random to explore different branches?
        # Let's try random selection from available.
        idx = np.random.choice(len(available))
        new_word = int(available[idx])
        
        codewords.append(new_word)
        
        # Update forbidden
        forbidden[new_word ^ ball_offsets] = True
        
        # Update available
        available = np.where(~forbidden)[0]
        
    return codewords
