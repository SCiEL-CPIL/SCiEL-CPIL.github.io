"""Exact frequency sketch using 1024-byte state for 400 items."""

# Each item i (0..399) gets a dedicated byte at index i in the state.
# Counts are capped at 255 to fit in a byte.

def update(state, item):
    """
    Increment the count for 'item' in the state.
    state: bytes of length 1024
    item: int in 0..399
    Returns new state as bytes of length 1024.
    """
    # Convert to bytearray for mutation
    ba = bytearray(state)
    # Increment the count for this item, capping at 255
    current = ba[item]
    if current < 255:
        ba[item] = current + 1
    return bytes(ba)

def query(state, item):
    """
    Return the exact frequency estimate for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns nonnegative int.
    """
    return state[item]
