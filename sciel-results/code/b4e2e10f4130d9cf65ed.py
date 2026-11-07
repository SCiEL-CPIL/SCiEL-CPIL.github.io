import struct

def update(state, item):
    """
    Updates the frequency counter for 'item' in the state.
    State layout:
    - Bytes 0..799: 400 little-endian uint16 counters for items 0..399.
    - Bytes 800..1023: Padding (unused).
    """
    # Ensure state is a bytearray for in-place modification
    # This copies the bytes, which is O(N) where N=1024. Very fast.
    ba = bytearray(state)
    
    # Calculate offset for the item's counter
    offset = item * 2
    
    # Read current count
    # '<H' is little-endian unsigned short (2 bytes)
    current_count = struct.unpack_from('<H', ba, offset)[0]
    
    # Increment
    new_count = current_count + 1
    
    # Write back
    struct.pack_into('<H', ba, offset, new_count)
    
    # Return the updated state as bytes
    return bytes(ba)

def query(state, item):
    """
    Queries the exact frequency of 'item' from the state.
    """
    offset = item * 2
    # Read the count directly from the state bytes
    return struct.unpack_from('<H', state, offset)[0]
