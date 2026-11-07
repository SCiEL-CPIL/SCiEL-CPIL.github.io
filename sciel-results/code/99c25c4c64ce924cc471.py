def update(state, item):
    # state is bytes
    # convert to bytearray
    buf = bytearray(state)
    
    # 1. Set bit in bitset (bytes 0-49)
    byte_idx = item // 8
    bit_idx = item % 8
    buf[byte_idx] |= (1 << bit_idx)
    
    # 2. Update counters (bytes 50-1021)
    # 2 rows, 243 buckets each
    # Bucket size 2 bytes (uint16)
    # Row 0 offset: 50
    # Row 1 offset: 50 + 243*2 = 536
    
    p1 = 1035493289
    p2 = 775374203
    
    # Row 0
    idx0 = (item * p1) % 243
    offset0 = 50 + idx0 * 2
    val0 = buf[offset0] | (buf[offset0+1] << 8)
    val0 += 1
    buf[offset0] = val0 & 0xFF
    buf[offset0+1] = (val0 >> 8) & 0xFF
    
    # Row 1
    idx1 = (item * p2 + 31) % 243
    offset1 = 536 + idx1 * 2
    val1 = buf[offset1] | (buf[offset1+1] << 8)
    val1 += 1
    buf[offset1] = val1 & 0xFF
    buf[offset1+1] = (val1 >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    # Check bitset
    byte_idx = item // 8
    bit_idx = item % 8
    if not (state[byte_idx] & (1 << bit_idx)):
        return 0
    
    # Query counters
    p1 = 1035493289
    p2 = 775374203
    
    idx0 = (item * p1) % 243
    offset0 = 50 + idx0 * 2
    val0 = state[offset0] | (state[offset0+1] << 8)
    
    idx1 = (item * p2 + 31) % 243
    offset1 = 536 + idx1 * 2
    val1 = state[offset1] | (state[offset1+1] << 8)
    
    return min(val0, val1)
