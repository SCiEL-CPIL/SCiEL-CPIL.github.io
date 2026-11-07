"""Direct-mapped exact counters + CM sketch overflow fallback for items 0..399."""

NUM_EXACT = 400
SKETCH_OFFSET = 400
NUM_ROWS = 8
NUM_COLS = 39  # 8 * 39 * 2 = 624 bytes

# Strong 32-bit mixing salts for each row
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
    """Bit-mixing hash to map item to column index."""
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update exact 8-bit counter (capped at 255)
    if ba[item] < 255:
        ba[item] += 1
    
    # Update CM sketch for overflow fallback
    for r in range(NUM_ROWS):
        idx = _hash_idx(item, r)
        off = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = ba[off] | (ba[off + 1] << 8)
        val += 1
        ba[off] = val & 0xFF
        ba[off + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    exact_val = state[item]
    
    if exact_val < 255:
        return float(exact_val)
    
    # Overflow: use CM sketch minimum
    min_val = 0xFFFF
    for r in range(NUM_ROWS):
        idx = _hash_idx(item, r)
        off = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val
    
    return float(min_val)
