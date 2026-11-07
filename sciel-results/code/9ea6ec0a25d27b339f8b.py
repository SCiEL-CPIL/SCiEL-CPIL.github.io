"""
Space-Saving Sketch implementation for memory-bounded streaming frequency estimation.
State layout (little-endian):
- Bytes 0-127: 32 uint32 counters
- Bytes 128-255: 32 uint32 item IDs
- Bytes 256-1023: Unused
"""

import struct
import sys

NUM_COUNTERS = 32
STATE_SIZE = 1024

def update(state, item):
    """
    Updates the Space-Saving sketch with a new item.
    Returns the new state as bytes.
    """
    # Unpack counters and items
    # 32 counters, 32 items
    counters = list(struct.unpack_from('<32I', state, 0))
    items = list(struct.unpack_from('<32I', state, 128))
    
    # Check if item is already tracked
    try:
        idx = items.index(item)
        counters[idx] += 1
    except ValueError:
        # Item not tracked. Find the minimum counter.
        min_val = min(counters)
        min_idx = counters.index(min_val)
        
        # Replace the minimum entry with the new item
        # The new count is min_val + 1
        counters[min_idx] = min_val + 1
        items[min_idx] = item

    # Pack back into state
    new_state = bytearray(STATE_SIZE)
    struct.pack_into('<32I', new_state, 0, *counters)
    struct.pack_into('<32I', new_state, 128, *items)
    return bytes(new_state)

def query(state, item):
    """
    Queries the Space-Saving sketch for the frequency estimate of an item.
    Returns a nonnegative integer estimate.
    """
    # Unpack items to check presence
    items = list(struct.unpack_from('<32I', state, 128))
    
    try:
        idx = items.index(item)
        # If tracked, return the exact count
        count = struct.unpack_from('<I', state, idx * 4)[0]
        return count
    except ValueError:
        # If not tracked, return the minimum count among all tracked items
        # This is the standard Space-Saving estimate for untracked items
        counters = list(struct.unpack_from('<32I', state, 0))
        return min(counters)

# EVOLVE-BLOCK-START
def update(state, item):
    return update(state, item)

def query(state, item):
    return query(state, item)
# EVOLVE-BLOCK-END
