"""Count-Min Sketch with 16 rows and 32 columns (16-bit counters).

Layout: 1024 bytes total.
- 16 rows x 32 counters x 2 bytes each.
  16 * 32 * 2 = 1024 bytes.

Rationale:
Increasing the number of rows (depth) in a Count-Min Sketch reduces the
probability that a high-frequency item collides with a low-frequency item
in *all* rows. Since the query returns the minimum, more rows provide
better protection against overestimation caused by collisions, which is
the primary source of error for low-frequency items in Zipf/Mixed streams.
Although the width is reduced to 32, the increased depth typically yields
a lower mean absolute error for this specific distribution profile.
"""

NUM_ROWS = 16
NUM_COLS = 32

# Per-row salt constants (odd 32-bit values for good mixing)
SALT = (
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x9e3779b9,
    0x85ebca77,
    0xc2b2ae3d,
    0x27d4eb31,
    0x165667b3,
    0x9e3779bb,
    0x85ebca79,
    0xc2b2ae3f,
    0x27d4eb33,
    0x165667b5,
    0x9e3779bd,
    0x85ebca7b
)

def _hash_idx(item, row):
    """Bit-mixing hash: multiply, XOR-shift, multiply, XOR-shift, take low 5 bits."""
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h & (NUM_COLS - 1)  # low 5 bits (0-31)

def update(state, item):
    ba = bytearray(state)
    
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    ba = state
    min_val = 0xFFFF
    
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
