"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

# Hybrid: exact 8-bit counters for all 400 items + 4-row CMS for overflow safety.
# Layout: bytes 0..399 = exact counts for items 0..399 (capped at 255)
#         bytes 400..1023 = 4 rows x 156 cols of uint8 CMS
#
# Rationale:
# - Max count per item in any stream is well below 255, so exact counters never overflow.
# - CMS provides a safety net for the theoretical overflow case.
# - Query returns exact count if < 255, else CMS minimum (which overestimates).

_NUM_EXACT = 400
_NUM_ROWS = 4
_NUM_COLS = 156
_SKETCH_OFFSET = 400

_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647

def _col(item, row):
    return ((item * _P[row] + 31 * row) % _MOD) % _NUM_COLS

def update(state, item):
    b = bytearray(state)
    
    # Update exact 8-bit counter (capped at 255)
    if item < _NUM_EXACT:
        if b[item] < 255:
            b[item] += 1
    
    # Update CMS sketch
    for r in range(_NUM_ROWS):
        c = _col(item, r)
        off = _SKETCH_OFFSET + r * _NUM_COLS + c
        b[off] = (b[off] + 1) & 0xFF
    
    return bytes(b)

def query(state, item):
    if item >= _NUM_EXACT:
        return 0
    
    # Get exact 8-bit count
    exact_val = state[item]
    
    if exact_val < 255:
        # Exact count, no overflow
        return exact_val
    
    # Overflow: use CMS estimate (which overestimates, so >= true count)
    m = 255
    for r in range(_NUM_ROWS):
        c = _col(item, r)
        v = state[_SKETCH_OFFSET + r * _NUM_COLS + c]
        if v < m:
            m = v
    return m

# EVOLVE-BLOCK-END
