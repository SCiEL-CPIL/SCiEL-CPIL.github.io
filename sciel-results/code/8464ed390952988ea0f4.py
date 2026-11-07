import struct

# Layout:
# 0..101: uint16 counts for items 0..50 (51 * 2 = 102 bytes)
# 102..178: uint8 counts for items 51..127 (77 * 1 = 77 bytes)
# 179..1023: CM Sketch for items 128..399
#   4 rows, 211 cols of uint8
#   4 * 211 = 844 bytes.
#   179 + 844 = 1023.
#   Total 1024 bytes.

P = (1035493289, 775374203, 1495862053, 2015145641)
CM_BASE = 179
CM_COLS = 211

def update(state, item):
    b = bytearray(state)
    
    if item <= 50:
        # uint16
        off = item * 2
        val = struct.unpack_from('<H', b, off)[0]
        struct.pack_into('<H', b, off, val + 1)
    elif item <= 127:
        # uint8
        off = 102 + (item - 51)
        val = b[off]
        b[off] = val + 1
    else:
        # CM Sketch
        for r in range(4):
            idx = (item * P[r]) % CM_COLS
            off = CM_BASE + r * CM_COLS + idx
            val = b[off]
            b[off] = val + 1
            
    return bytes(b)

def query(state, item):
    if item <= 50:
        off = item * 2
        val = struct.unpack_from('<H', state, off)[0]
        return float(val)
    elif item <= 127:
        off = 102 + (item - 51)
        return float(state[off])
    else:
        min_val = 255
        for r in range(4):
            idx = (item * P[r]) % CM_COLS
            off = CM_BASE + r * CM_COLS + idx
            val = state[off]
            if val < min_val:
                min_val = val
        return float(min_val)
