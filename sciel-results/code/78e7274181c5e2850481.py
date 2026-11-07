import array

def update(state, item):
    """
    Updates the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int
    returns: bytes (1024 bytes)
    
    Strategy:
    Use the 1024-byte state to store an exact frequency count for each of the 400 items.
    Each count is stored as a 16-bit unsigned integer (uint16).
    Item i is stored at index i in a conceptual array of 512 uint16s.
    Total bytes used: 400 * 2 = 800 bytes.
    Remaining 224 bytes are unused (remain 0).
    Max possible count is 1024, which fits in uint16 (max 65535).
    """
    # Convert state bytes to a mutable array of unsigned shorts
    # 'H' is 2 bytes unsigned short
    # We assume native byte order is consistent between update and query
    counts = array.array('H', state)
    
    # Increment the count for the given item
    # item is guaranteed to be in 0..399
    counts[item] += 1
    
    # Convert back to bytes
    return counts.tobytes()

def query(state, item):
    """
    Queries the sketch state for the frequency estimate of an item.
    state: bytes (1024 bytes)
    item: int
    returns: int (non-negative frequency estimate)
    """
    # Convert state bytes to an array of unsigned shorts
    counts = array.array('H', state)
    
    # Return the exact count
    return counts[item]
