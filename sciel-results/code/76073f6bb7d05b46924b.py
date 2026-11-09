"""
Hybrid frequency sketch: exact uint8 counters for items 0-127,
4-row x 224-slot uint8 Count-Min sketch for items 128-399.
"""
import struct

# Constants
ROWS = 4
SLOTS = 224  # 4 * 224 = 896 bytes for the sketch
EXACT_COUNT = 128  # items 0..127 get exact counters
TOTAL_ITEMS = 400

# Hash constants for the sketch rows (same primes as seed)
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647  # 2^31 - 1


def _sketch_index(item, row):
    """Compute the slot index for a given item and row in the 224-slot sketch."""
    # Offset items 128-399 to 0-271 to spread them in the hash
    x = item - 128
    return ((x * _P[row] + 31 * row) % _MOD) % SLOTS


def update(state, item):
    """Update the sketch state with a new item."""
    state = bytearray(state)
    
    if item < EXACT_COUNT:
        # Exact counter: increment, cap at 255
        idx = item
        if state[idx] < 255:
            state[idx] += 1
    else:
        # Count-Min sketch for items 128-399
        base = EXACT_COUNT  # sketch starts at byte 128
        for row in range(ROWS):
            slot = _sketch_index(item, row)
            byte_idx = base + row * SLOTS + slot
            if state[byte_idx] < 255:
                state[byte_idx] += 1
    
    return bytes(state)


def query(state, item):
    """Query the estimated frequency of an item."""
    if item < EXACT_COUNT:
        # Exact counter
        return state[item]
    else:
        # Count-Min sketch: take minimum over rows
        base = EXACT_COUNT
        min_val = 256
        for row in range(ROWS):
            slot = _sketch_index(item, row)
            byte_idx = base + row * SLOTS + slot
            val = state[byte_idx]
            if val < min_val:
                min_val = val
        return min_val
