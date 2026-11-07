import struct

# Precompute hash parameters
P1 = 1035493289
P2 = 775374203
P3 = 1495862053
P4 = 2015145641
MOD = 2147483647
WIDTH = 64
DEPTH = 4

def get_indices(item, row):
    # Returns index in [0, WIDTH)
    if row == 0:
        return ((item * P1 + 31 * 0) % MOD) % WIDTH
    elif row == 1:
        return ((item * P2 + 31 * 1) % MOD) % WIDTH
    elif row == 2:
        return ((item * P3 + 31 * 2) % MOD) % WIDTH
    else:
        return ((item * P4 + 31 * 3) % MOD) % WIDTH

def update(state, item):
    # State layout:
    # Bytes 0-63: Bitset (512 bits)
    # Bytes 64-575: Count-Min Sketch (4 rows x 64 cols x uint16)
    
    new_state = bytearray(state)
    
    # 1. Update Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    new_state[byte_idx] |= mask
    
    # 2. Update CM Sketch
    # Counters are uint16 LE
    for row in range(DEPTH):
        idx = get_indices(item, row)
        # Offset in state: 64 + row * (WIDTH * 2) + idx * 2
        offset = 64 + row * (WIDTH * 2) + idx * 2
        # Read current value
        val = struct.unpack_from('<H', new_state, offset)[0]
        # Increment
        val += 1
        # Write back
        struct.pack_into('<H', new_state, offset, val)
        
    return bytes(new_state)

def query(state, item):
    # 1. Check Bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0.0
        
    # 2. Query CM Sketch
    min_val = float('inf')
    for row in range(DEPTH):
        idx = get_indices(item, row)
        offset = 64 + row * (WIDTH * 2) + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
            
    return float(min_val)
