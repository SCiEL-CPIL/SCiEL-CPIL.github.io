import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 17
    MIN_DIST = 6
    MAX_SPACE = 1 << N
    
    # 1. Initialize with RM(1,4) code (32 words)
    # These are affine functions on F_2^4 embedded in 17 bits.
    # The RM(1,4) code has parameters [16, 5, 8].
    # We embed it in 17 bits by appending 0.
    # Min distance remains 8.
    code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit (index 16) is 0
            code.append(word)
            
    # 2. Greedy linear extension
    # We try to add vectors v to the code such that min distance remains >= 6.
    # Current code size is 32 (2^5). We want to reach 256 (2^8).
    # So we need to find 3 more linearly independent vectors.
    
    target_size = 256
    
    for v in range(1, MAX_SPACE):
        if len(code) >= target_size:
            break
        if time.time() > deadline:
            break
            
        # Check if v can be added
        valid = True
        # We can break early if any conflict found.
        for c in code:
            if (v ^ c).bit_count() < MIN_DIST:
                valid = False
                break
                
        if valid:
            # Add v and all c^v
            # We must iterate over the OLD code to generate new words.
            new_words = []
            for c in code:
                new_words.append(c ^ v)
            code.extend(new_words)
            
    return code
