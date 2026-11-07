def update(state, item):
    b = bytearray(state)
    
    # Bitset: bytes 0-49 (50 bytes = 400 bits)
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    
    # If not yet seen, mark it and increment seen_count
    if not (b[byte_idx] & bit_mask):
        b[byte_idx] |= bit_mask
        # Increment seen_count at bytes 50-51 (uint16)
        sc = int.from_bytes(b[50:52], 'little')
        b[50:52] = (sc + 1).to_bytes(2, 'little')
    
    # Increment global counter at bytes 52-53 (uint16)
    gc = int.from_bytes(b[52:54], 'little')
    b[52:54] = (gc + 1).to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    # Check bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    
    if not (state[byte_idx] & bit_mask):
        return 0
    
    # Seen item: return global_count // seen_count
    seen_count = int.from_bytes(state[50:52], 'little')
    global_count = int.from_bytes(state[52:54], 'little')
    
    if seen_count == 0:
        return 0
    
    return global_count // seen_count
