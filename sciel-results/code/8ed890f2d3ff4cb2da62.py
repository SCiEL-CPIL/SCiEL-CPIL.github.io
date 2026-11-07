def update(state, item):
    b = bytearray(state)
    
    # 1. Update Bitset (bytes 0-49)
    # item is 0..399
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    b[byte_idx] |= mask
    
    # 2. Update Counter (bytes 50-1023)
    # 974 counters of uint8
    # Hash function
    h = (item * 1035493289 + 31) % 2147483647
    idx = h % 974
    
    offset = 50 + idx
    # Use wrapping addition to handle potential overflow safely
    b[offset] = (b[offset] + 1) & 0xFF
    
    return bytes(b)

def query(state, item):
    # 1. Check Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0
    
    # 2. Read Counter
    h = (item * 1035493289 + 31) % 2147483647
    idx = h % 974
    
    offset = 50 + idx
    val = state[offset]
    
    return val
