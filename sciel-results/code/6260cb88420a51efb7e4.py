import struct

def update(state, item):
    # State layout:
    # Bytes 0-127: 1024-bit bitset for items 0..1023 (we use 0..399)
    # Bytes 128-131: Total count (uint32 LE)
    # Bytes 132-135: Distinct count (uint32 LE)
    
    # We assume state is 1024 bytes.
    # We only modify the first 136 bytes, rest remains 0 (or whatever was there, but we copy).
    
    # To be safe and fast, we create a new bytearray
    new_state = bytearray(state) # Copy existing state
    
    # Update bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    # Check if already set
    if not (new_state[byte_idx] & mask):
        new_state[byte_idx] |= mask
        # Increment distinct count
        distinct_count = struct.unpack('<I', new_state[132:136])[0]
        distinct_count += 1
        struct.pack_into('<I', new_state, 132, distinct_count)
    
    # Increment total count
    total_count = struct.unpack('<I', new_state[128:132])[0]
    total_count += 1
    struct.pack_into('<I', new_state, 128, total_count)
    
    return bytes(new_state)

def query(state, item):
    # Parse state
    # Bitset is at 0-127
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0.0
    
    # Item seen
    total_count = struct.unpack('<I', state[128:132])[0]
    distinct_count = struct.unpack('<I', state[132:136])[0]
    
    if distinct_count == 0:
        return 0.0
        
    return float(total_count) / float(distinct_count)
