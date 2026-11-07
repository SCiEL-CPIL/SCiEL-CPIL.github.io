"""Count-Min Sketch with bit-mixing hash for better distribution.

Layout: 1024 bytes total.
- 8 rows x 64 counters x 2 bytes each (16-bit counters).
  8 * 64 * 2 = 1024 bytes.

Hashing: Uses a multiplicative bit-mixing hash (MurmurHash3 finalizer style)
to avoid the systematic bias of (i*p) % M % 64 where small i values
cluster in predictable low bits.
"""

NUM_ROWS = 8
NUM_COLS = 64

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
)

def _hash_idx(item, row):
    """Bit-mixing hash: multiply, XOR-shift, multiply, XOR-shift, take low 6 bits."""
    h = (item + 1) * SALT[row]  # avoid 0 mapping to 0
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h & (NUM_COLS - 1)  # low 6 bits

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
