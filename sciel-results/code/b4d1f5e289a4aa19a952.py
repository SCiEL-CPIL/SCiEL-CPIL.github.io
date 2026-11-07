import struct

# Layout:
# Bytes 0..799: 16-bit exact counters for items 0..399 (400 items * 2 bytes)
# Bytes 800..1023: 224 bytes = 4 rows * 28 cols * 2 bytes (16-bit CMS counters)

NUM_EXACT = 400
SKETCH_OFFSET = 800
NUM_ROWS = 4
NUM_COLS = 28

# Salt constants for CM sketch hashing
SALT = (
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
)

def _hash_idx(item, row):
    """Bit-mixing hash to map (item, row) to a column index in 0..NUM_COLS-1."""
    h = (item + 1) * SALT[row]
    h &= 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update exact 16-bit counter (capped at 65535)
    off = item * 2
    val = ba[off] | (ba[off + 1] << 8)
    if val < 65535:
        val += 1
        ba[off] = val & 0xFF
        ba[off + 1] = (val >> 8) & 0xFF
    
    # Update CM sketch for all 4 rows
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        if val > 65535:
            val = 65535
        
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    # Get exact 16-bit count
    off = item * 2
    exact_val = state[off] | (state[off + 1] << 8)
    
    if exact_val < 65535:
        # Exact count, no overflow
        return float(exact_val)
    
    # Overflow: use CM sketch estimate (min over rows, which overestimates true count)
    min_val = 0xFFFF
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = state[offset] | (state[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
    
    return float(min_val)
