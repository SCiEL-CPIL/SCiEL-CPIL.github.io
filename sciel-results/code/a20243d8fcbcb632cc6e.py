"""
Exact frequency counting using uint16 counters.
State layout (1024 bytes):
  Bytes 0-799: 400 little-endian uint16 counters for items 0-399.
  Bytes 800-1023: Unused padding (zeros).

Update: Increment the uint16 counter for the given item.
Query: Return the exact count from the uint16 counter.

This approach provides exact counts for all items, as the maximum possible
count (1024) fits within a uint16 (max 65535). This results in zero
estimation error, maximizing the search score.
"""
# EVOLVE-BLOCK-START

def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    
    Args:
        state: bytes, the current state (1024 bytes).
        item: int, the item to update (0..399).
        
    Returns:
        bytes, the new state.
    """
    buf = bytearray(state)
    
    # Calculate offset for the item's uint16 counter
    offset = item * 2
    
    # Read current value (little-endian uint16)
    # byte 0 is LSB, byte 1 is MSB
    low = buf[offset]
    high = buf[offset + 1]
    current_val = low + (high << 8)
    
    # Increment
    new_val = current_val + 1
    
    # Write back (little-endian uint16)
    buf[offset] = new_val & 0xFF
    buf[offset + 1] = (new_val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """
    Query the state for the frequency estimate of the given item.
    
    Args:
        state: bytes, the current state (1024 bytes).
        item: int, the item to query (0..399).
        
    Returns:
        int, the exact frequency count.
    """
    # Calculate offset for the item's uint16 counter
    offset = item * 2
    
    # Read value (little-endian uint16)
    low = state[offset]
    high = state[offset + 1]
    
    return low + (high << 8)

# EVOLVE-BLOCK-END
