"""
Strategy: Direct Frequency Table (Exact Histogram)
Since the item range is small (0..399) and the state size is sufficient (1024 bytes),
we can store the exact count for each item using 2 bytes (uint16) per item.
Total storage: 400 * 2 = 800 bytes.
Remaining 224 bytes are unused padding.
This results in 0 estimation error.
"""

# EVOLVE-BLOCK-START
def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    State layout: 400 uint16 counters, little-endian, followed by padding.
    """
    # Convert to mutable bytearray
    s = bytearray(state)
    
    # Calculate offset for this item's counter
    # Each item uses 2 bytes (uint16)
    offset = item * 2
    
    # Read current count (little-endian uint16)
    low_byte = s[offset]
    high_byte = s[offset + 1]
    current_count = low_byte + (high_byte << 8)
    
    # Increment
    new_count = current_count + 1
    
    # Write back (little-endian uint16)
    s[offset] = new_count & 0xFF
    s[offset + 1] = (new_count >> 8) & 0xFF
    
    return bytes(s)

def query(state, item):
    """
    Query the exact count for the given item.
    """
    # Calculate offset
    offset = item * 2
    
    # Read count (little-endian uint16)
    low_byte = state[offset]
    high_byte = state[offset + 1]
    
    return low_byte + (high_byte << 8)
# EVOLVE-BLOCK-END
