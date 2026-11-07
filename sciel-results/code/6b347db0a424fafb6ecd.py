import struct

# EVOLVE-BLOCK-START

# Full-domain Count-Min sketch: 8 rows x 128 columns of uint16 (2 bytes each)
# Total: 8 * 128 * 2 = 2048 bytes... wait, that's too much.
# Let's recalculate: 1024 bytes / 2 bytes per counter = 512 counters total
# 8 rows x 64 columns x 2 bytes = 1024 bytes. That works!
# Or: 4 rows x 128 columns x 2 bytes = 1024 bytes.
# Let's go with 8 rows x 64 columns of uint16.

CM_ROWS = 8
CM_COLS = 64
# 8 * 64 * 2 = 1024 bytes exactly

# 8 distinct large odd hash constants
P_CONSTANTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
]

MOD = 2147483647  # 2^31 - 1

def _hash_idx(item, row):
    p = P_CONSTANTS[row]
    return ((item * p + 31 * row) % MOD) % CM_COLS

def update(state, item):
    s = bytearray(state)
    for row in range(CM_ROWS):
        idx = _hash_idx(item, row)
        pos = (row * CM_COLS + idx) * 2
        val = struct.unpack_from('<H', s, pos)[0]
        val += 1
        struct.pack_into('<H', s, pos, val)
    return bytes(s)

def query(state, item):
    min_val = 1024  # Max possible count (1024 items total)
    for row in range(CM_ROWS):
        idx = _hash_idx(item, row)
        pos = (row * CM_COLS + idx) * 2
        val = struct.unpack_from('<H', state, pos)[0]
        if val < min_val:
            min_val = val
    return min_val

# EVOLVE-BLOCK-END
