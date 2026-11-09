import struct

PRIMES = [1035493289, 775374203, 1495862053, 2015145641, 1604569091, 16807, 48271, 22698433]
M = 2147483647

def update(state, item):
    # Unpack 8 rows of 64 uint16s
    counters = struct.unpack('<64H' * 8, state)
    # Flatten to list for easier manipulation? Or just index directly.
    # State is 1024 bytes.
    # We need to increment 8 specific counters.
    
    # It's faster to work with a list of ints
    c = list(counters)
    
    for i in range(8):
        p = PRIMES[i]
        s = 31 * (i + 1) # Distinct offsets
        idx = ((item * p + s) % M) % 64
        row_offset = i * 64
        c[row_offset + idx] += 1
        
    return struct.pack('<64H' * 8, *c)

def query(state, item):
    counters = struct.unpack('<64H' * 8, state)
    c = list(counters)
    min_val = float('inf')
    
    for i in range(8):
        p = PRIMES[i]
        s = 31 * (i + 1)
        idx = ((item * p + s) % M) % 64
        row_offset = i * 64
        val = c[row_offset + idx]
        if val < min_val:
            min_val = val
            
    return min_val
