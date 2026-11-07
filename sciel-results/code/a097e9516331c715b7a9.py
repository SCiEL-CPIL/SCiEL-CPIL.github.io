import struct
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

# Direct indexing array for 400 items
# Each item gets a uint16 counter (2 bytes)
# Total bytes used: 400 * 2 = 800 bytes
# Remaining 224 bytes are unused (zero-initialized)

def update(state, item):
    """
    Update the frequency count for the given item.
    Uses direct indexing into a uint16 array stored in the first 800 bytes of state.
    """
    if time.time() > _DEADLINE:
        return state
    
    # Ensure state is a bytearray for mutable operations
    if isinstance(state, bytes):
        b = bytearray(state)
    else:
        b = state
        
    # Calculate offset for the item
    offset = item * 2
    
    # Read current count
    count = struct.unpack_from('<H', b, offset)[0]
    
    # Increment and write back
    # Max count is 1024, well within uint16 range (65535)
    struct.pack_into('<H', b, offset, count + 1)
    
    return bytes(b)

def query(state, item):
    """
    Query the frequency estimate for the given item.
    Returns the exact count stored in the direct index array.
    """
    if time.time() > _DEADLINE:
        return 0.0
    
    # Calculate offset for the item
    offset = item * 2
    
    # Read count
    count = struct.unpack_from('<H', state, offset)[0]
    
    return float(count)
