"""
Combined streaming frequency sketch:
- 400-bit membership bitset (50 bytes) to return 0 for unseen items.
- 4 rows of 121 uint16 counters (968 bytes) for Count-Min sketch estimates.
- Total: 50 + 968 = 1018 bytes (6 bytes padding to reach 1024).

The bitset eliminates collision noise for unseen items (significant for uniform streams
where most items have low or zero frequency). The CM sketch with width 121 and depth 4
provides better estimates for seen items than a single row, by taking the minimum of
4 independent hash rows to reduce overestimation from collisions.

Hashing: h = ((item * p + 31 * row) % 2147483647) % 121
Primes from the seed: (1035493289, 775374203, 1495862053, 2015145641)
"""

def update(state, item):
    b = bytearray(state)
    
    # Update bitset: set bit for item
    byte_idx = item // 8
    bit_idx = item % 8
    b[byte_idx] |= (1 << bit_idx)
    
    # Update 4 counters
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    width = 121  # Number of counters per row
    counter_base = 50  # Counters start after the 50-byte bitset
    
    for row in range(4):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % width
        
        # Offset: row * width * 2 + idx * 2
        offset = counter_base + row * (width * 2) + idx * 2
        
        # Read uint16
        val = b[offset] | (b[offset + 1] << 8)
        val += 1
        # Write back
        b[offset] = val & 0xFF
        b[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    
    # Check bitset
    byte_idx = item // 8
    bit_idx = item % 8
    if not (b[byte_idx] & (1 << bit_idx)):
        return 0
    
    # Item is seen, return min of 4 counters
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    width = 121
    counter_base = 50
    
    min_val = 65535  # Max uint16
    
    for row in range(4):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % width
        
        offset = counter_base + row * (width * 2) + idx * 2
        
        val = b[offset] | (b[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
