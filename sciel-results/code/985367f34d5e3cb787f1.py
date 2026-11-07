import struct

def update(state, item):
    # Ensure state is a bytearray for mutation
    s = bytearray(state)
    
    # Each item has a dedicated 16-bit counter.
    # Item i is stored at byte offset i * 2.
    offset = item * 2
    
    # Unpack the current 16-bit little-endian value
    count = struct.unpack_from('<H', s, offset)[0]
    
    # Increment the count
    count += 1
    
    # Clamp to max uint16 just in case, though unlikely to be reached
    if count > 65535:
        count = 65535
        
    # Pack the new value back into the state
    struct.pack_into('<H', s, offset, count)
    
    return bytes(s)

def query(state, item):
    # Each item has a dedicated 16-bit counter.
    # Item i is stored at byte offset i * 2.
    offset = item * 2
    
    # Unpack the 16-bit little-endian value
    count = struct.unpack_from('<H', state, offset)[0]
    
    return count
