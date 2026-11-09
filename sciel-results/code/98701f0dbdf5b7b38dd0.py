import struct

# Fixed-Window Exact Counter for items 0..255 (1 byte each) + 4-row x 192-slot uint16 CMS for items 256..399
# State layout:
#   bytes 0..255: exact counters for items 0..255 (uint8)
#   bytes 256..1023: 4 rows of 192 uint16 counters for items 256..399
#   Total: 256 + 4*192*2 = 256 + 1536... wait, 4*192*2 = 1536 which exceeds 768.
#   Let's recalculate: 1024 - 256 = 768 bytes for sketch.
#   If using uint16 counters: 768 / 2 = 384 counters total.
#   With 4 rows: 384 / 4 = 96 slots per row.
#   So: 4 rows of 96 uint16 counters.

EXACT_N = 256  # Items 0..255 get exact counters
SKETCH_ROWS = 4
SKETCH_W = 96  # 768 bytes / 2 bytes per counter / 4 rows = 96
PRIME = 2147483647
_P = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute indices for sketch (only for items 256..399)
# _SKETCH_INDICES[item - 256][row] = index
_SKETCH_INDICES = [[0] * SKETCH_ROWS for _ in range(400 - EXACT_N)]
for i in range(400 - EXACT_N):
    item = EXACT_N + i
    for r in range(SKETCH_ROWS):
        _SKETCH_INDICES[i][r] = ((item * _P[r] + 31 * r) % PRIME) % SKETCH_W

def update(state, item):
    buf = bytearray(state)
    if item < EXACT_N:
        # Exact counter: 1 byte per item
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        # Sketch counter: uint16
        i = item - EXACT_N
        idxs = _SKETCH_INDICES[i]
        for r in range(SKETCH_ROWS):
            idx = idxs[r]
            offset = EXACT_N + r * SKETCH_W * 2 + idx * 2
            cur = int.from_bytes(buf[offset:offset+2], 'little')
            if cur < 65535:
                buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    return bytes(buf)

def query(state, item):
    if item < EXACT_N:
        return state[item]
    else:
        i = item - EXACT_N
        idxs = _SKETCH_INDICES[i]
        min_val = 65535
        for r in range(SKETCH_ROWS):
            idx = idxs[r]
            offset = EXACT_N + r * SKETCH_W * 2 + idx * 2
            val = int.from_bytes(state[offset:offset+2], 'little')
            if val < min_val:
                min_val = val
        return min_val
