def update(state, item):
    b = bytearray(state)
    
    # 1. Update Bitset (bytes 0-49)
    # item is 0..399
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    b[byte_idx] |= mask
    
    # 2. Update Counter (bytes 50-1023)
    # 487 counters of uint16
    # Hash function
    h = (item * 1035493289 + 31) % 2147483647
    idx = h % 487
    
    offset = 50 + idx * 2
    val = int.from_bytes(b[offset:offset+2], 'little')
    val += 1
    b[offset:offset+2] = val.to_bytes(2, 'little')
    
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
    idx = h % 487
    
    offset = 50 + idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    
    return val
