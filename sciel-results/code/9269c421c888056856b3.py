import numpy as np

# Constants
P = [1035493289, 775374203, 1495862053, 2015145641]
MOD_LARGE = 2147483647
WIDTH = 128
ROWS = 4

def update(state, item):
    # Create a writable copy of the state
    counters = np.array(np.frombuffer(state, dtype=np.uint16).reshape(ROWS, WIDTH), dtype=np.uint16)
    
    for r in range(ROWS):
        p = P[r]
        idx = ((item * p + 31 * r) % MOD_LARGE) % WIDTH
        counters[r, idx] += 1
        
    return counters.tobytes()

def query(state, item):
    counters = np.frombuffer(state, dtype=np.uint16).reshape(ROWS, WIDTH)
    
    min_val = 65535
    for r in range(ROWS):
        p = P[r]
        idx = ((item * p + 31 * r) % MOD_LARGE) % WIDTH
        val = counters[r, idx]
        if val < min_val:
            min_val = val
            
    return int(min_val)
