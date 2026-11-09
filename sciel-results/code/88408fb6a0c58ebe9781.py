import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    RADIUS = MIN_DIST - 1  # 5
    
    # Precompute all vectors in GF(2)^N with weight <= RADIUS
    # These are the offsets for the Hamming ball
    ball_offsets = []
    for i in range(2**N):
        if i.bit_count() <= RADIUS:
            ball_offsets.append(i)
    
    ball_offsets = np.array(ball_offsets, dtype=np.uint32)
    
    # Forbidden array
    forbidden = np.zeros(2**N, dtype=bool)
    
    codewords = []
    
    # Start with 0
    codewords.append(0)
    # Mark forbidden zone around 0
    forbidden[ball_offsets] = True
    
    while time.time() < deadline:
        # Find available candidates
        available = np.where(~forbidden)[0]
        
        if len(available) == 0:
            break
        
        # Select the first available candidate (lowest weight/value)
        # This tends to produce a good packing
        new_word = int(available[0])
        
        # Add to codewords
        codewords.append(new_word)
        
        # Update forbidden zone
        # new_word ^ offsets
        new_forbidden = new_word ^ ball_offsets
        forbidden[new_forbidden] = True
        
        # Optional: Break if we reached a large number? 
        # The loop will break when no more are available.
        
    return codewords
