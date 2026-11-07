"""Two-tier frequency sketch: exact counters for hot items + large CM for cold."""
import struct

HOT_SIZE = 10          # exact counters for items 0..9
HOT_BYTES = HOT_SIZE * 4  # 40 bytes (uint32, safe for max count 1024)

# Remaining bytes: 1024 - 40 = 984 -> 492 uint16 counters
# CM: 12 rows x 41 cols = 492 counters
ROWS = 12
COLS = 41
CM_BASE_BYTE = HOT_BYTES  # offset where CM counters start
CM_COUNTERS = ROWS * COLS  # 492

# Verify: 40 + 492*2 = 40 + 984 = 1024 bytes
assert HOT_BYTES + CM_COUNTERS * 2 <= 1024

# Hash parameters for 12 rows
# Use distinct odd multipliers and offsets for good distribution mod COLS
P_VALS = [
    1035493289, 775374203, 1495862053, 2015145641,
    1103515245, 1234567891, 987654321, 876543219,
    765432197, 654321983, 543219871, 432198757
]
Q_VALS = [31, 17, 53, 79, 11, 41, 67, 23, 37, 59, 83, 101]

# Precompute byte offsets for each cold item (10..399) and each row
# LUT[item][row] = byte offset of the uint16 counter
_LUT = [[0] * ROWS for _ in range(400)]

for item in range(HOT_SIZE, 400):
    for r in range(ROWS):
        col = (item * P_VALS[r] + Q_VALS[r]) % COLS
        # Counter index within CM = r * COLS + col
        # Byte offset = CM_BASE_BYTE + (r * COLS + col) * 2
        _LUT[item][r] = CM_BASE_BYTE + (r * COLS + col) * 2


def update(state, item):
    state = bytearray(state)
    
    if item < HOT_SIZE:
        # Exact uint32 counter
        off = item * 4
        val = struct.unpack_from('<I', state, off)[0]
        val += 1
        struct.pack_into('<I', state, off, val)
    else:
        # Increment 12 uint16 CM counters
        for r in range(ROWS):
            off = _LUT[item][r]
            val = struct.unpack_from('<H', state, off)[0]
            val += 1
            struct.pack_into('<H', state, off, val)
    
    return bytes(state)


def query(state, item):
    if item < HOT_SIZE:
        # Exact count
        off = item * 4
        return float(struct.unpack_from('<I', state, off)[0])
    else:
        # Min over 12 rows of uint16 counters
        min_val = 65535
        for r in range(ROWS):
            off = _LUT[item][r]
            val = struct.unpack_from('<H', state, off)[0]
            if val < min_val:
                min_val = val
        return float(min_val)
