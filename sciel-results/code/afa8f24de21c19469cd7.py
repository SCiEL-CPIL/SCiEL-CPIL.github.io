import struct

# 8 rows, 64 columns, uint16
# Total 512 counters.

# Primes for hashing
# Need 8 distinct primes
PRIMES = [
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    19349663, 
    16777619, 
    15949, 
    1000003
]

def _get_index(item, row):
    # (item * p + offset) % 64
    # Using modulo 64 is fast
    p = PRIMES[row]
    # Add a small offset to avoid correlation if p is even? p is prime (odd).
    # item * p + row * some_const
    return (item * p + row * 12345) % 64

def update(state, item):
    # Unpack 512 unsigned shorts
    counters = list(struct.unpack('<512H', state))
    
    for r in range(8):
        idx = _get_index(item, r)
        offset = r * 64 + idx
        counters[offset] += 1
        
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = list(struct.unpack('<512H', state))
    
    min_val = float('inf')
    for r in range(8):
        idx = _get_index(item, r)
        offset = r * 64 + idx
        val = counters[offset]
        if val < min_val:
            min_val = val
            
    return float(min_val)
