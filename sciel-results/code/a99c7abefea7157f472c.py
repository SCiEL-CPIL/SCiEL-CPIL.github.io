import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 17
    MIN_DIST = 6
    MAX_SPACE = 1 << N
    
    # 1. Build the 256-word linear code using greedy linear extension
    # Start with RM(1,4) embedded in 17 bits (32 words, min dist 8)
    code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            code.append(word)
    
    # Greedy linear extension to reach 256 words
    target_size = 256
    for v in range(1, MAX_SPACE):
        if len(code) >= target_size:
            break
        if time.time() > deadline - 15:
            break
        valid = True
        for c in code:
            if (v ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            new_words = [c ^ v for c in code]
            code.extend(new_words)
    
    code_np = np.array(code, dtype=np.int32)
    code_size = len(code)
    
    # 2. Vectorized nonlinear extension using forbidden zone marking
    # Precompute all XOR masks with popcount < 6 (i.e., 1 to 5)
    masks = []
    for mask_val in range(1, MAX_SPACE):
        pc = mask_val.bit_count()
        if pc < MIN_DIST:
            masks.append(mask_val)
    masks_np = np.array(masks, dtype=np.int32)
    
    # Forbidden zone: boolean array of size 2^17
    forbidden = np.zeros(MAX_SPACE, dtype=bool)
    
    # Mark forbidden zones for all current codewords
    for c in code_np:
        forbidden[c ^ masks_np] = True
        forbidden[c] = True
    
    # Get un-forbidden candidates
    unforbidden = np.where(~forbidden)[0]
    
    # Greedily add nonlinear words
    for candidate in unforbidden:
        if time.time() > deadline - 3:
            break
        if forbidden[candidate]:
            continue
        
        # Add this word
        code.append(int(candidate))
        
        # Mark its forbidden zone
        forbidden[candidate ^ masks_np] = True
        forbidden[candidate] = True
    
    return code
