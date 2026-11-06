import time
_start_time = time.time()
_TIMEOUT = 320.0 # Safety margin

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9
    
    c0 = 0
    c1 = 0
    c2 = 0
    h = 0
    runs = 1
    prev = -1
    
    # Single pass
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        
        if v != prev:
            runs += 1
        prev = v
        
        h = (h * 31 + v) % 1000000007
            
    ideal = n / 3.0
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Normalize hash to [0, 1)
    h_norm = h / 1000000007.0
    
    # We want to maximize priority.
    # 1. Minimize dev (primary).
    # 2. Maximize runs (secondary, favors "random" over "blocky").
    # 3. Tie break with hash (tertiary).
    
    return -dev + runs * 1e-6 - h_norm * 1e-9
