"""Direct-mapped exact counters + compact CM sketch fallback for items 0..399."""

NUM_EXACT = 400
SKETCH_OFFSET = 400
NUM_ROWS = 4
NUM_COLS = 156  # 4 * 156 * 2 = 1248 > 624, but we only have 624 bytes -> 4*78*2=624
# Actually: 624 / 2 / 4 = 78 columns per row
NUM_COLS = 78

# Precomputed hash indices for speed: hash_idx[row][item]
# Use simple multiplicative hash with different primes
_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = (1 << 32) - 1

def _build_hash_table():
    """Precompute hash indices for all items and rows."""
    table = []
    for r in range(NUM_ROWS):
        row = []
        p = _PRIMES[r]
        for item in range(400):
            h = ((item + 1) * p + 31 * r) & 0xFFFFFFFF
            row.append(h % NUM_COLS)
        table.append(row)
    return table

_HASH_TABLE = _build_hash_table()

def update(state, item):
    ba = bytearray(state)
    
    # Update exact 8-bit counter (capped at 255)
    if ba[item] < 255:
        ba[item] += 1
    
    # Update CM sketch for overflow fallback
    for r in range(NUM_ROWS):
        idx = _HASH_TABLE[r][item]
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
        idx = _HASH_TABLE[r][item]
        off = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val
    
    return float(min_val)
