"""Hybrid frequency sketch: exact 8-bit counters + 16-row CM sketch for overflow.

Layout: 1024 bytes total.
- Bytes 0..399: 8-bit exact counters for items 0..399 (capped at 255).
- Bytes 400..703: 16 rows x 19 cols of 16-bit CM sketch (304 bytes).
- Bytes 704..1023: reserved (320 bytes, unused for now).

Strategy:
- For items with true count < 256, the exact byte gives perfect estimate.
- For items with true count >= 256, fall back to 16-row CM sketch.
  More rows = lower collision probability = tighter overestimate.
"""

NUM_EXACT = 400
NUM_ROWS = 16
NUM_COLS = 19
SKETCH_OFFSET = 400
MAX_EXACT = 255

SALT = (
    0x9e3779b9, 0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f,
    0x165667b1, 0x85ebca77, 0xc2b2ae3d, 0x27d4eb31,
    0x9e3779c9, 0x85ebca7b, 0xc2b2ae45, 0x27d4eb3f,
    0x165667b9, 0x85ebca7f, 0xc2b2ae4d, 0x27d4eb39,
)


def _hash_idx(item, row):
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS


_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(NUM_ROWS):
        row_indices.append(_hash_idx(item, r))
    _IDX_TABLE.append(row_indices)


def update(state, item):
    ba = bytearray(state)

    exact_val = ba[item]
    if exact_val < MAX_EXACT:
        ba[item] = exact_val + 1

    item_indices = _IDX_TABLE[item]
    for r in range(NUM_ROWS):
        idx = item_indices[r]
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        if val > 0xFFFF:
            val = 0xFFFF
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    exact_val = state[item]

    if exact_val < MAX_EXACT:
        return exact_val

    item_indices = _IDX_TABLE[item]
    min_val = 0xFFFF
    for r in range(NUM_ROWS):
        idx = item_indices[r]
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = state[offset] | (state[offset + 1] << 8)
        if val < min_val:
            min_val = val

    return min_val
