import struct

# 4 rows x 128 columns of uint16 counters = 1024 bytes
ROWS = 4
COLS = 128
# Multipliers chosen for good low-bit distribution on items 0..399
PRIMES = (
    1035493289,
    775374203,
    1495862053,
    2015145641,
)
# Offsets to decorrelate rows
OFFSETS = (
    31, 199, 555, 1029,
)

def _get_index(item, row):
    """
    Calculate the index for a given item and row.
    """
    p = PRIMES[row]
    off = OFFSETS[row]
    # 32-bit truncation then modulo COLS
    h = (item * p + off) & 0xFFFFFFFF
    return h % COLS

def update(state, item):
    """
    Update the frequency sketch for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns new state as bytes of length 1024.
    """
    # Convert to mutable buffer
    buf = bytearray(state)

    for r in range(ROWS):
        idx = _get_index(item, r)
        byte_offset = (r * COLS + idx) * 2  # Each counter is 2 bytes (uint16)

        # Read current value (little-endian uint16)
        val = struct.unpack_from('<H', buf, byte_offset)[0]

        # Increment
        val += 1

        # Write back
        struct.pack_into('<H', buf, byte_offset, val)

    return bytes(buf)

def query(state, item):
    """
    Query the frequency estimate for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns nonnegative int (minimum estimate across rows).
    """
    min_val = float('inf')

    for r in range(ROWS):
        idx = _get_index(item, r)
        byte_offset = (r * COLS + idx) * 2

        # Read value
        val = struct.unpack_from('<H', state, byte_offset)[0]

        if val < min_val:
            min_val = val

    # If no value was found (should not happen with 4 rows), return 0
    if min_val == float('inf'):
        return 0

    return min_val
