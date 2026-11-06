import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 17
    min_dist = 6
    max_size = 340 # Upper bound from Brouwer
    
    codewords = [0]
    
    # Optimization: Pre-calculate bit counts for small numbers? No, numbers are 17 bits.
    # Just use bit_count.
    
    for candidate in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Check distance to all existing codewords
        valid = True
        for cw in codewords:
            if (candidate ^ cw).bit_count() < min_dist:
                valid = False
                break
                
        if valid:
            codewords.append(candidate)
            if len(codewords) >= max_size:
                break
                
    return codewords
