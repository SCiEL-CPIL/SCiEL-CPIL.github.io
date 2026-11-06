import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0  # Seconds
    N = 18
    MIN_DISTANCE = 6
    mask = (1 << N) - 1
    
    # Quadratic residues modulo 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Candidate base vectors for the cyclic generator
    # We represent the base vector as an integer bitmask of length 18.
    # Bit i corresponds to position i.
    
    candidates = []
    
    # Candidate 1: 1s at Q and 0, 0 at pos 17
    u1 = 0
    for i in range(17):
        if i == 0 or i in Q:
            u1 |= (1 << i)
    candidates.append(u1)
    
    # Candidate 2: 1s at Q and 0, 1 at pos 17
    u2 = u1 | (1 << 17)
    candidates.append(u2)
    
    # Candidate 3: 1s at Q only, 0 at pos 0, 1 at pos 17
    u3 = 0
    for i in Q:
        u3 |= (1 << i)
    u3 |= (1 << 17)
    candidates.append(u3)
    
    # Candidate 4: 1s at Q only, 1 at pos 0, 0 at pos 17
    u4 = 0
    for i in Q:
        u4 |= (1 << i)
    u4 |= (1 << 0)
    candidates.append(u4)

    # Candidate 5: 1s at Q only, 0 at pos 0, 0 at pos 17
    u5 = 0
    for i in Q:
        u5 |= (1 << i)
    candidates.append(u5)
        
    # Candidate 6: 1s at Non-Residues and 0, 0 at pos 17
    NR = set(range(1, 17)) - Q
    u6 = 0
    for i in range(17):
        if i == 0 or i in NR:
            u6 |= (1 << i)
    candidates.append(u6)

    best_words = []
    best_size = 0

    for u in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Generate 9 basis vectors by cyclic right shifts
        basis = []
        current = u & mask
        for _ in range(9):
            basis.append(current)
            # Rotate right by 1
            current = ((current >> 1) | ((current & 1) << (N - 1))) & mask
        
        # Generate all 512 codewords
        words = [0] * 512
        # Use Gray code or simple iteration? Simple iteration is fine for 512.
        for i in range(512):
            w = 0
            # Unroll loop for speed? 9 iterations is small.
            for j in range(9):
                if (i >> j) & 1:
                    w ^= basis[j]
            words[i] = w
        
        # Verify pairwise distances
        # Optimization: Check if min distance is at least MIN_DISTANCE
        # We can early exit if we find a pair with distance < MIN_DISTANCE
        valid = True
        # Sort words to potentially help cache? Not really.
        # Just check all pairs.
        # 512 * 511 / 2 = 130,576 pairs.
        # Each pair: XOR and bit_count.
        # This is fast.
        for i in range(512):
            if not valid:
                break
            wi = words[i]
            for j in range(i + 1, 512):
                d = (wi ^ words[j]).bit_count()
                if d < MIN_DISTANCE:
                    valid = False
                    break
        
        if valid:
            # Ensure distinctness (linear code basis ensures distinctness if rank is 9)
            # Rank check: if basis vectors are linearly independent, all 512 are distinct.
            # Cyclic shifts of a vector of weight 9 in length 18 are likely independent.
            # We can verify distinctness by checking if len(set(words)) == 512
            if len(set(words)) == 512:
                if len(words) > best_size:
                    best_size = len(words)
                    best_words = words
                    # We found a valid 512-word code. 
                    # Is it possible to have more?
                    # A_2(18, 6) is known to be 512? 
                    # If 512 is the max, we can return immediately.
                    # The prompt says "512-word ... reference", implying it's the target.
                    # Let's return immediately to save time, assuming 512 is the max or close enough.
                    # Actually, let's just return the first valid 512-word code.
                    return words

    # If no 512-word code found, return the best found (likely empty or smaller)
    # Fallback to the original Reed-Muller construction if all fail
    if not best_words:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    return best_words
