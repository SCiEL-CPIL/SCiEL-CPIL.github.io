import struct

# Primes for hashing
PRIMES = [
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    1000000007, 
    1000000009, 
    1000000033, 
    1000000087
]

ROWS = 8
COLS = 64
MOD = 2147483647

def update(state, item):
    # Unpack 512 uint16s
    counters = list(struct.unpack('<512H', state))
    
    for r in range(ROWS):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % MOD) % COLS
        offset = r * COLS + idx
        counters[offset] += 1
        
    # Pack back
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = list(struct.unpack('<512H', state))
    
    min_est = float('inf')
    for r in range(ROWS):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % MOD) % COLS
        offset = r * COLS + idx
        c = counters[offset]
        if c < min_est:
            min_est = c
            
    return min_est
