"""
Memory-bounded streaming frequency sketch.

Uses a Count-Min Sketch with 4 rows x 64 columns of uint32 counters (1024 bytes total).
The hash functions use the same parameters as the seed to ensure compatibility,
but we optimize the update and query paths for speed.

Design:
- 4 rows of 64 little-endian uint32 counters
- Hash: ((item * p + 31 * row) % 2147483647) % 64
- Query: min across all rows
- Update: increment all 4 counters
"""
import struct
import time

# Hash parameters (same as seed)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647
NUM_ROWS = 4
NUM_COLS = 64
STATE_SIZE = 1024  # 4 * 64 * 4 bytes

# Precompute hash offsets for all 400 items across all 4 rows
# _HASH_TABLE[item][row] = column index
_HASH_TABLE = [[0] * NUM_ROWS for _ in range(400)]
for _item in range(400):
    for _row in range(NUM_ROWS):
        _HASH_TABLE[_item][_row] = ((_item * PRIMES[_row] + 31 * _row) % MOD) % NUM_COLS

# Precompute byte offsets for each (row, col) position
# Each counter is 4 bytes (uint32 LE)
def _byte_offset(row, col):
    return row * NUM_COLS * 4 + col * 4

# Precompute byte offsets for each item and row
_BYTE_OFFSETS = [[0] * NUM_ROWS for _ in range(400)]
for _item in range(400):
    for _row in range(NUM_ROWS):
        _col = _HASH_TABLE[_item][_row]
        _BYTE_OFFSETS[_item][_row] = _byte_offset(_row, _col)


def update(state, item):
    """Update the sketch state with one item insertion."""
    # Ensure state is exactly 1024 bytes
    if len(state) != STATE_SIZE:
        state = state.ljust(STATE_SIZE, b'\x00')[:STATE_SIZE]
    
    offsets = _BYTE_OFFSETS[item]
    # Work with a mutable copy
    buf = bytearray(state)
    for row in range(NUM_ROWS):
        off = offsets[row]
        # Read current uint32 LE
        val = struct.unpack_from('<I', buf, off)[0]
        val += 1
        # Write back
        struct.pack_into('<I', buf, off, val)
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item."""
    if len(state) < STATE_SIZE:
        state = state.ljust(STATE_SIZE, b'\x00')
    
    offsets = _BYTE_OFFSETS[item]
    min_val = 0xFFFFFFFF
    for row in range(NUM_ROWS):
        off = offsets[row]
        val = struct.unpack_from('<I', state, off)[0]
        if val < min_val:
            min_val = val
    return min_val
