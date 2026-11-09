"""Memory-bounded streaming frequency sketches: Fixed-Window Exact + CM tail."""
import struct
import array

# Layout (1024 bytes total):
# Bytes 0..255:   exact counters for items 0..255 (1 byte each, capped at 255)
# Bytes 256..1023: 4 rows x 96 slots x uint16 = 4 * 96 * 2 = 768 bytes
# Total: 256 + 768 = 1024. Correct.

SKETCH_ROWS = 4
SKETCH_SLOTS = 96  # uint16 slots per row
SKETCH_OFFSET = 256  # byte offset where sketch starts
SKETCH_ROW_BYTES = SKETCH_SLOTS * 2  # 192 bytes per row

# Hash constants for the tail sketch (items 256..399)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
HASH_MOD = 2147483647


def update(state, item):
    """
    Update the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes (1024 bytes)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return b'\x00' * 1024

    buf = bytearray(state)

    if item < 256:
        # Exact counter
        idx = item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
        # else: cap at 255
    else:
        # Tail item (256..399): use CM sketch with uint16 counters
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            # Hash: ((item * p + 31 * row) % HASH_MOD) % SKETCH_SLOTS
            h = ((item * p + 31 * row) % HASH_MOD) % SKETCH_SLOTS

            # Offset for uint16 in row r, slot h
            # Row r starts at SKETCH_OFFSET + r * SKETCH_ROW_BYTES
            # Slot h is at offset + h * 2
            offset = SKETCH_OFFSET + row * SKETCH_ROW_BYTES + h * 2

            # Read uint16 LE
            val = struct.unpack_from('<H', buf, offset)[0]

            # Increment (cap at 65535)
            if val < 65535:
                val += 1
            struct.pack_into('<H', buf, offset, val)

    return bytes(buf)


def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative frequency estimate (int or float)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0

    if item < 256:
        # Exact count
        return state[item]
    else:
        # CM sketch query: min over rows
        min_val = 65535
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            h = ((item * p + 31 * row) % HASH_MOD) % SKETCH_SLOTS

            offset = SKETCH_OFFSET + row * SKETCH_ROW_BYTES + h * 2

            val = struct.unpack_from('<H', state, offset)[0]

            if val < min_val:
                min_val = val

        return min_val
