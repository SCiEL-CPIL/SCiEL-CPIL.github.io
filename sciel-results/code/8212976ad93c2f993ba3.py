import struct

def update(state, item):
    # State layout:
    # 0-127: Bloom Filter (1024 bits)
    # 128-131: Global Counter (uint32) - Not strictly needed for CM, but good for debugging? 
    #          Actually, CM doesn't need global counter.
    # 128-1023: Count-Min Sketch (4 rows x 56 cols x 4 bytes = 896 bytes)
    # Total: 128 + 896 = 1024 bytes.

    new_state = bytearray(state)
    
    # 1. Update Bloom Filter
    # Use 3 hash functions
    # h1 = (item * 1035493289) % 1024
    # h2 = (item * 775374203) % 1024
    # h3 = (item * 1495862053) % 1024
    
    h1 = (item * 1035493289) % 1024
    h2 = (item * 775374203) % 1024
    h3 = (item * 1495862053) % 1024
    
    # Set bits
    for h in [h1, h2, h3]:
        byte_idx = h // 8
        bit_idx = h % 8
        new_state[byte_idx] |= (1 << bit_idx)
        
    # 2. Update Count-Min Sketch
    # 4 rows, 56 columns
    # p values
    ps = [1035493289, 775374203, 1495862053, 2015145641]
    
    for r in range(4):
        col = (item * ps[r] + r * 31) % 56
        offset = 128 + r * (56 * 4) + col * 4
        # Read uint32
        val = struct.unpack_from('<I', new_state, offset)[0]
        val += 1
        struct.pack_into('<I', new_state, offset, val)
        
    return bytes(new_state)

def query(state, item):
    # 1. Check Bloom Filter
    h1 = (item * 1035493289) % 1024
    h2 = (item * 775374203) % 1024
    h3 = (item * 1495862053) % 1024
    
    in_bloom = True
    for h in [h1, h2, h3]:
        byte_idx = h // 8
        bit_idx = h % 8
        if not (state[byte_idx] & (1 << bit_idx)):
            in_bloom = False
            break
            
    if not in_bloom:
        return 0
        
    # 2. Query Count-Min Sketch
    ps = [1035493289, 775374203, 1495862053, 2015145641]
    min_val = float('inf')
    
    for r in range(4):
        col = (item * ps[r] + r * 31) % 56
        offset = 128 + r * (56 * 4) + col * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
            
    return min_val
