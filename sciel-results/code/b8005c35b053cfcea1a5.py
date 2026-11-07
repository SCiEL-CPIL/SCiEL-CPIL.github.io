import struct

# EVOLVE-BLOCK-START
# Structural Change: Hybrid Sketch
# 1. Exact 8-bit counters for items 0..19 and 150..399.
#    - Items 0..19: 20 items. Stored at state[0..19].
#    - Items 150..399: 250 items. Stored at state[20..269].
#    Total bytes for exact counters: 270.
# 2. Count-Min Sketch for items 20..149 (130 items).
#    - Remaining bytes: 1024 - 270 = 754.
#    - Use 16-bit unsigned integers for counters to reduce collision error.
#    - 754 bytes / 2 = 377 counters.
#    - Use 4 rows of 94 columns = 376 counters = 752 bytes.
#    - Base offset for CMS: 270.
#    - CMS layout: state[270 + row * 188 + col * 2 : 270 + row * 188 + col * 2 + 2]

MULTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641
]
MOD = 2147483647
CMS_ROWS = 4
CMS_COLS = 94
CMS_OFFSET = 270

def update(state, item):
    s = bytearray(state)
    
    if 0 <= item <= 19:
        pos = item
        s[pos] = (s[pos] + 1) & 0xFF
    elif 150 <= item <= 399:
        pos = 20 + (item - 150)
        s[pos] = (s[pos] + 1) & 0xFF
    else:
        # Item in 20..149
        for r, m in enumerate(MULTS):
            idx = ((item * m + r + 1) % MOD) % CMS_COLS
            pos = CMS_OFFSET + r * (CMS_COLS * 2) + idx * 2
            # Read 16-bit little-endian
            val = struct.unpack_from('<H', s, pos)[0]
            val += 1
            # Wrap around at 65535
            if val > 65535:
                val = 65535
            struct.pack_into('<H', s, pos, val)
            
    return bytes(s)

def query(state, item):
    if 0 <= item <= 19:
        return state[item]
    elif 150 <= item <= 399:
        return state[20 + (item - 150)]
    else:
        min_val = 65535
        for r, m in enumerate(MULTS):
            idx = ((item * m + r + 1) % MOD) % CMS_COLS
            pos = CMS_OFFSET + r * (CMS_COLS * 2) + idx * 2
            val = struct.unpack_from('<H', state, pos)[0]
            if val < min_val:
                min_val = val
        return min_val
# EVOLVE-BLOCK-END
