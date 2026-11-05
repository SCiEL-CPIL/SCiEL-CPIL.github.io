import time
import numpy as np

N = 18
MIN_DIST = 6
DEADLINE = 90.0

def construct():
    start_time = time.time()
    
    # Hardcoded generator matrix for the [18,9,6] code
    # This is the extended QR code of length 18 (or equivalent)
    # I will use a known generator matrix for the [18,9,6] code.
    # If I don't have it, I will use the greedy approach.
    
    # For now, I will use the greedy approach with a good seed.
    # Seed: Reed-Muller RM(1,4) [16,5,8] code, extended to 18 bits.
    
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
            
    code = np.array(rm_words, dtype=np.uint32)
    
    # Fast greedy expansion
    # Use a deadline guard
    while time.time() - start_time < DEADLINE - 5.0:
        # Generate random word
        new_word = np.random.randint(0, 2**N, dtype=np.uint32)
        
        # Check if it can be added
        # Use vectorized check for speed
        xors = code ^ new_word
        # Check popcount
        # For small code, loop is fine
        valid = True
        for x in xors:
            if x.bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            code = np.append(code, new_word)
            
    return [int(x) for x in code]
