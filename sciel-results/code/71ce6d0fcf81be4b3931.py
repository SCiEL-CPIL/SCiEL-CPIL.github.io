"""Hybrid exact/sketch frequency estimator."""
import struct

# Layout: bytes [0:896] = 4 rows x 224 uint32 (sketch)
#          bytes [896:1024] = 32 entries x (2 bytes item_id+1, 2 bytes count) = 128 bytes
SKETCH_ROWS = 4
SKETCH_COLS = 224
EXACT_SLOTS = 32
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647
SKETCH_BYTES = SKETCH_ROWS * SKETCH_COLS * 4  # 896

def _hash(item, row):
    return ((item * PRIMES[row] + 31 * row) % MOD) % SKETCH_COLS

def update(state, item):
    state = bytearray(state)
    
    # Update sketch
    for r in range(SKETCH_ROWS):
        idx = _hash(item, r)
        off = r * SKETCH_COLS * 4 + idx * 4
        val = struct.unpack_from('<I', state, off)[0]
        val = min(val + 1, 0xFFFFFFFF)
        struct.pack_into('<I', state, off, val)
    
    # Update exact table
    base = SKETCH_BYTES
    # Find if item is in exact table (stored as item+1, 0 means empty)
    found = -1
    for i in range(EXACT_SLOTS):
        off = base + i * 4
        eid = struct.unpack_from('<H', state, off)[0]
        if eid == item + 1:
            found = i
            break
    
    if found >= 0:
        off = base + found * 4 + 2
        cnt = struct.unpack_from('<H', state, off)[0]
        cnt = min(cnt + 1, 0xFFFF)
        struct.pack_into('<H', state, off, cnt)
    else:
        # Find empty slot or evict
        empty = -1
        min_idx = -1
        min_cnt = 0xFFFF
        for i in range(EXACT_SLOTS):
            off = base + i * 4
            eid = struct.unpack_from('<H', state, off)[0]
            if eid == 0:  # empty
                if empty < 0:
                    empty = i
                continue
            cnt = struct.unpack_from('<H', state, off + 2)[0]
            if cnt < min_cnt:
                min_cnt = cnt
                min_idx = i
        
        if empty >= 0:
            slot = empty
        else:
            # Get sketch estimate for this item
            sk_est = 0xFFFFFFFF
            for r in range(SKETCH_ROWS):
                idx = _hash(item, r)
                off = r * SKETCH_COLS * 4 + idx * 4
                val = struct.unpack_from('<I', state, off)[0]
                if val < sk_est:
                    sk_est = val
            if sk_est > min_cnt:
                slot = min_idx
            else:
                return bytes(state)
        
        off = base + slot * 4
        struct.pack_into('<H', state, off, item + 1)
        struct.pack_into('<H', state, off + 2, 1)
    
    return bytes(state)

def query(state, item):
    state = bytes(state)
    base = SKETCH_BYTES
    
    # Check exact table
    for i in range(EXACT_SLOTS):
        off = base + i * 4
        eid = struct.unpack_from('<H', state, off)[0]
        if eid == item + 1:
            cnt = struct.unpack_from('<H', state, off + 2)[0]
            return float(cnt)
    
    # Fall back to sketch
    est = 0xFFFFFFFF
    for r in range(SKETCH_ROWS):
        idx = _hash(item, r)
        off = r * SKETCH_COLS * 4 + idx * 4
        val = struct.unpack_from('<I', state, off)[0]
        if val < est:
            est = val
    return float(est)
