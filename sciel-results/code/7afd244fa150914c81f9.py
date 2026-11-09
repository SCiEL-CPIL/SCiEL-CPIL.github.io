"""Memory-bounded streaming frequency sketches: hybrid exact + CM sketch."""
import struct
import time

# Layout:
#   Bytes 0..255: 256 exact uint8 counters for items 0..255
#   Bytes 256..1023: 768 bytes = 4 rows × 192 slots of uint16 counters for items 256..399

_ROWS = 4
_SLOTS = 192
_EXACT_SIZE = 256
_SKETCH_OFFSET = 256
_TOTAL_SIZE = 1024

# Hash primes for the CM sketch (same as seed, adapted for modulo 192)
_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)


def _sketch_index(item, row):
    """Compute the column index for a given item and row in the CM sketch."""
    p = _PRIMES[row]
    return ((item * p + 31 * row) % 2147483647) % _SLOTS


def update(state, item):
    """Update the sketch state with a new item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return state  # Should not happen, but guard

    state = bytearray(state)

    if 0 <= item < _EXACT_SIZE:
        # Exact counter: increment, saturate at 255
        idx = item
        if state[idx] < 255:
            state[idx] += 1
    else:
        # CM sketch for items 256..399
        # Map item to a reduced key for hashing (item - 256 gives 0..143)
        key = item - 256
        for row in range(_ROWS):
            col = ((key * _PRIMES[row] + 31 * row) % 2147483647) % _SLOTS
            byte_offset = _SKETCH_OFFSET + row * _SLOTS * 2 + col * 2
            # Read uint16 little-endian
            val = state[byte_offset] | (state[byte_offset + 1] << 8)
            if val < 65535:
                val += 1
            state[byte_offset] = val & 0xFF
            state[byte_offset + 1] = (val >> 8) & 0xFF

    return bytes(state)


def query(state, item):
    """Query the estimated frequency of an item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return 0

    if 0 <= item < _EXACT_SIZE:
        # Exact count
        return float(state[item])
    else:
        # CM sketch: return minimum across rows
        key = item - 256
        min_val = 65535
        for row in range(_ROWS):
            col = ((key * _PRIMES[row] + 31 * row) % 2147483647) % _SLOTS
            byte_offset = _SKETCH_OFFSET + row * _SLOTS * 2 + col * 2
            val = state[byte_offset] | (state[byte_offset + 1] << 8)
            if val < min_val:
                min_val = val
        return float(min_val)
