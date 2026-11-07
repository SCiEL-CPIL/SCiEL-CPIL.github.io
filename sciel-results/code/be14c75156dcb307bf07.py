"""Program using exact membership bitset and global average frequency estimate."""
import struct

def update(state, item):
    # State layout:
    # Bytes 0-63: 512-bit bitset for items 0..511 (we use 0..399)
    # Bytes 64-67: Total count (uint32 LE)
    # Bytes 68-71: Distinct count (uint32 LE)
    
    # Parse current state
    # We assume state is 1024 bytes
    # Extract bitset
    bitset_bytes = bytearray(state[0:64])
    
    # Extract counters
    total_count = struct.unpack('<I', state[64:68])[0]
    distinct_count = struct.unpack('<I', state[68:72])[0]
    
    # Update logic
    # Check if item is already seen
    # Item is int 0..399
    # Bit index is item
    byte_idx = item // 8
    bit_idx = item % 8
    
    if bit_idx < 8:
        mask = 1 << bit_idx
        if not (bitset_bytes[byte_idx] & mask):
            # Item not seen before
            bitset_bytes[byte_idx] |= mask
            distinct_count += 1
    
    # Increment total count
    total_count += 1
    
    # Construct new state
    new_state = bytearray(1024)
    new_state[0:64] = bitset_bytes
    new_state[64:68] = struct.pack('<I', total_count)
    new_state[68:72] = struct.pack('<I', distinct_count)
    
    return bytes(new_state)

def query(state, item):
    # Parse state
    bitset_bytes = state[0:64]
    total_count = struct.unpack('<I', state[64:68])[0]
    distinct_count = struct.unpack('<I', state[68:72])[0]
    
    # Check membership
    byte_idx = item // 8
    bit_idx = item % 8
    
    mask = 1 << bit_idx
    
    if not (bitset_bytes[byte_idx] & mask):
        # Item not seen
        return 0.0
    
    # Item seen
    if distinct_count == 0:
        return 0.0
        
    # Return global average frequency
    return float(total_count) / float(distinct_count)

# EVOLVE-BLOCK-START
# The functions above are the implementation. 
# The task requires keeping function names and signatures.
# The current program wraps them in imports. I will replace the entire block.
# EVOLVE-BLOCK-END
