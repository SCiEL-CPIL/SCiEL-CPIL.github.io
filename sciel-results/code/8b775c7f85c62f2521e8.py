def update(state, item):
    """
    Updates the state with a new item occurrence.
    Uses 2 bytes (uint16) per item for exact counting.
    Item range: 0..399. Total bytes: 400 * 2 = 800.
    Max count: 1024, which fits in uint16 (max 65535).
    """
    buf = bytearray(state)
    # Calculate offset for this item
    idx = item * 2
    
    # Read current count
    # Little-endian uint16
    val = buf[idx] | (buf[idx + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    buf[idx] = val & 0xFF
    buf[idx + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """
    Queries the estimated frequency of an item.
    Returns the exact count stored in the state.
    """
    idx = item * 2
    # Little-endian uint16
    return state[idx] | (state[idx + 1] << 8)
