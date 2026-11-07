"""Program using 1024-bit exact membership bitset and global average frequency estimate."""
import struct

def update(state, item):
    # State layout:
    # Bytes 0-127: 1024-bit bitset for items 0..1023 (we use 0..399)
    # Bytes 128-131: Total count (uint32 LE)
    # Bytes 132-135: Distinct count (uint32 LE)
    # Rest is padding
    
    # Reuse state bytes to avoid allocation overhead
    new_state = bytearray(state)
    
    # Extract counters
    total_count = struct.unpack('<I', state[128:132])[0]
    distinct_count = struct.unpack('<I', state[132:136])[0]
    
    # Update logic
    # Item is int 0..399
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    # Check if item is already seen
    if not (new_state[byte_idx] & mask):
        # Item not seen before
        new_state[byte_idx] |= mask
        distinct_count += 1
    
    # Increment total count
    total_count += 1
    
    # Write back counters
    new_state[128:132] = struct.pack('<I', total_count)
    new_state[132:136] = struct.pack('<I', distinct_count)
    
    return bytes(new_state)

def query(state, item):
    # Parse state
    # Bitset is at 0-127
    total_count = struct.unpack('<I', state[128:132])[0]
    distinct_count = struct.unpack('<I', state[132:136])[0]
    
    # Check membership
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        # Item not seen
        return 0.0
    
    # Item seen
    if distinct_count == 0:
        return 0.0
        
    # Return global average frequency
    return float(total_count) / float(distinct_count)
