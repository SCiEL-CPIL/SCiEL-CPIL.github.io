import struct
import time

# Precompute hash parameters for 8 rows of CMS
# We use 1024 bytes total: 8 rows * 64 counters * 2 bytes (uint16) = 1024 bytes
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    2147483629,
    1000000007,
    999999937
]
Q_CONSTS = [
    31, 137, 523, 1987, 7919, 30011, 117497, 461891
]
MERSenne = 2147483647  # 2^31 - 1
WIDTH = 64

def update(state, item):
    """
    Updates the frequency counter for 'item' in the state.
    State layout: 8 rows of 64 little-endian uint16 counters (1024 bytes total).
    """
    ba = bytearray(state)
    
    for r in range(8):
        p = PRIMES[r]
        q = Q_CONSTS[r]
        # High-quality hash to 0..63
        idx = ((item * p + q) % MERSenne) % WIDTH
        offset = r * 128 + idx * 2
        
        # Read uint16, increment, write back
        current = struct.unpack_from('<H', ba, offset)[0]
        new_val = current + 1
        struct.pack_into('<H', ba, offset, new_val)
    
    return bytes(ba)

def query(state, item):
    """
    Queries the estimated frequency of 'item' using the minimum over all rows.
    """
    min_count = 65535  # Max uint16 value
    
    for r in range(8):
        p = PRIMES[r]
        q = Q_CONSTS[r]
        idx = ((item * p + q) % MERSenne) % WIDTH
        offset = r * 128 + idx * 2
        count = struct.unpack_from('<H', state, offset)[0]
        if count < min_count:
            min_count = count
    
    return min_count
