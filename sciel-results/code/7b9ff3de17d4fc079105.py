import struct

# Exact frequency counting: 400 items, each with a 1-byte counter
# State layout: 400 bytes (one per item) + 624 bytes of padding (zeros)
# Total: 1024 bytes

def update(state, item):
    """
    Updates the exact frequency counter for 'item' in the state.
    State layout: 400 bytes of uint8 counters (one per item 0..399) + padding.
    """
    ba = bytearray(state)
    # item is 0..399, so direct index
    ba[item] = (ba[item] + 1) & 0xFF
    return bytes(ba)

def query(state, item):
    """
    Queries the exact frequency of 'item'.
    """
    return state[item]
