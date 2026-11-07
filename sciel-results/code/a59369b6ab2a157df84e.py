import struct

# Precompute hash indices for all 400 items and 4 rows
# 4 rows, 128 columns of 16-bit counters
# Total bytes: 4 * 128 * 2 = 1024

# Primes for hashing (odd numbers to ensure good distribution of lower bits)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
# Salts to decorrelate rows
SALTS = (0, 12345, 67890, 112233)

# Precompute indices: hash_indices[item][row] = index in [0, 127]
_hash_indices = [[0] * 4 for _ in range(400)]
for item in range(400):
    for r in range(4):
        # Use bitwise AND with 127 to get index in 0..127
        # This is equivalent to % 128 for positive numbers
        _hash_indices[item][r] = (item * PRIMES[r] + SALTS[r]) & 127

def update(state, item):
    counters = bytearray(state)
    
    for r in range(4):
        idx = _hash_indices[item][r]
        offset = r * 256 + idx * 2
        
        # Read 16-bit little-endian
        val = counters[offset] | (counters[offset + 1] << 8)
        
        # Increment
        val += 1
        
        # Write back 16-bit little-endian
        counters[offset] = val & 0xFF
        counters[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(counters)

def query(state, item):
    counters = bytearray(state)
    
    min_val = 65535
    for r in range(4):
        idx = _hash_indices[item][r]
        offset = r * 256 + idx * 2
        
        val = counters[offset] | (counters[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
