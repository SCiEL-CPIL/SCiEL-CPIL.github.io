"""
Pure 10-row x 64-column Count-Min Sketch with 16-bit counters.
State layout (1024 bytes):
  - 10 rows * 64 columns * 2 bytes (uint16 LE) = 1280 bytes... wait, 10*64*2 = 1280 > 1024.
  - Correction: 8 rows * 64 columns * 2 bytes = 1024 bytes.
  - Let's use 8 rows x 64 cols x uint16.
  
Strategy:
- Pure CM sketch for all 400 items.
- 8 rows, 64 columns.
- Each counter is 16 bits (max 65535), which is more than enough for counts up to 1024.
- Hash function: ((item * p + 31 * row) % 2147483647) % 64
- Query returns the minimum of the 8 counters.
"""

# EVOLVE-BLOCK-START

# Parameters
_NUM_ROWS = 8
_NUM_COLS = 64
_MOD = 2147483647  # 2^31 - 1

# Primes for hashing - using distinct large primes
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
]

# Pre-compute CM indices for all 400 items
# For each item, store list of byte_offsets
_CM_OFFSETS = {}
for item in range(400):
    offsets = []
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        # Each counter is 2 bytes (uint16 LE)
        byte_offset = (row * _NUM_COLS + idx) * 2
        offsets.append(byte_offset)
    _CM_OFFSETS[item] = offsets


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    # CM sketch: increment all 8 counters
    for byte_off in _CM_OFFSETS[item]:
        val = buf[byte_off] | (buf[byte_off + 1] << 8)
        val = (val + 1) & 0xFFFF
        buf[byte_off] = val & 0xFF
        buf[byte_off + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative float."""
    # CM sketch: return minimum of all 8 counters
    min_val = 0xFFFF
    for byte_off in _CM_OFFSETS[item]:
        val = state[byte_off] | (state[byte_off + 1] << 8)
        if val < min_val:
            min_val = val
    return float(min_val)
# EVOLVE-BLOCK-END
