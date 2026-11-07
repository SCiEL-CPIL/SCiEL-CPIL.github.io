import struct

def update(state, item):
    state = bytearray(state)
    
    # Layout:
    # Bytes 0-49:   Bitset for 400 items (50 bytes)
    # Bytes 50-1023: 4 rows of 64 counters (uint16), total 4*64*2 = 512 bytes
    # Total used: 562 bytes. Remaining 462 bytes are unused (zero).
    
    # 1. Update Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    state[byte_idx] |= mask
    
    # 2. Update Count-Min Sketch
    # We use 4 rows of 64 uint16 counters.
    # This gives us a "minimum of 4" estimator which reduces collision bias.
    # Width W=64. Total items N=1024.
    # Expected collision noise per row ~ N/W = 16.
    # Min of 4 Poisson(16) is significantly lower than 16.
    # Using uint16 is safe since max count per bucket is bounded by N=1024, which fits in uint16 (max 65535).
    
    base_offset = 50
    row_size = 64
    # Primes for hashing
    p1 = 1035493289
    p2 = 775374203
    p3 = 1495862053
    p4 = 2015145641
    mod = 2147483647
    
    # Row 0
    idx0 = ((item * p1 + 31 * 0) % mod) % row_size
    off0 = base_offset + 0 * row_size * 2 + idx0 * 2
    val0 = int.from_bytes(state[off0:off0+2], 'little')
    state[off0:off0+2] = (val0 + 1).to_bytes(2, 'little')
    
    # Row 1
    idx1 = ((item * p2 + 31 * 1) % mod) % row_size
    off1 = base_offset + 1 * row_size * 2 + idx1 * 2
    val1 = int.from_bytes(state[off1:off1+2], 'little')
    state[off1:off1+2] = (val1 + 1).to_bytes(2, 'little')
    
    # Row 2
    idx2 = ((item * p3 + 31 * 2) % mod) % row_size
    off2 = base_offset + 2 * row_size * 2 + idx2 * 2
    val2 = int.from_bytes(state[off2:off2+2], 'little')
    state[off2:off2+2] = (val2 + 1).to_bytes(2, 'little')
    
    # Row 3
    idx3 = ((item * p4 + 31 * 3) % mod) % row_size
    off3 = base_offset + 3 * row_size * 2 + idx3 * 2
    val3 = int.from_bytes(state[off3:off3+2], 'little')
    state[off3:off3+2] = (val3 + 1).to_bytes(2, 'little')
    
    return bytes(state)

def query(state, item):
    # 1. Check Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0
    
    # 2. Query Count-Min Sketch
    base_offset = 50
    row_size = 64
    p1 = 1035493289
    p2 = 775374203
    p3 = 1495862053
    p4 = 2015145641
    mod = 2147483647
    
    idx0 = ((item * p1 + 31 * 0) % mod) % row_size
    off0 = base_offset + 0 * row_size * 2 + idx0 * 2
    val0 = int.from_bytes(state[off0:off0+2], 'little')
    
    idx1 = ((item * p2 + 31 * 1) % mod) % row_size
    off1 = base_offset + 1 * row_size * 2 + idx1 * 2
    val1 = int.from_bytes(state[off1:off1+2], 'little')
    
    idx2 = ((item * p3 + 31 * 2) % mod) % row_size
    off2 = base_offset + 2 * row_size * 2 + idx2 * 2
    val2 = int.from_bytes(state[off2:off2+2], 'little')
    
    idx3 = ((item * p4 + 31 * 3) % mod) % row_size
    off3 = base_offset + 3 * row_size * 2 + idx3 * 2
    val3 = int.from_bytes(state[off3:off3+2], 'little')
    
    return min(val0, val1, val2, val3)
