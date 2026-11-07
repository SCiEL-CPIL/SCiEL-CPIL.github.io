"""Domain-adaptive hybrid sketch: exact hot counters + widened CM sketch."""

# EVOLVE-BLOCK-START

# Precompute indices for the Count-Min sketch
# 4 rows, 128 columns. Total width 512 bytes for the CM part.
# We use 16-bit counters for the CM part to handle higher counts if needed, 
# though for 1024 items, 8 bits might suffice, but 16 bits is safer and 
# fits in the budget.
# Layout:
# Bytes 0-19:   20 uint16 counters for items 0..19 (hot items)
# Bytes 20-531: 4 rows of 128 uint16 counters for CM sketch (512 bytes)
# Bytes 532-1023: Unused (zero)

# Primes for hashing
p0 = 1035493289
p1 = 775374203
p2 = 1495862053
p3 = 2015145641

def _cm_index(item, row):
    if row == 0:
        return ((item * p0 + 31 * 0) % 2147483647) % 128
    elif row == 1:
        return ((item * p1 + 31 * 1) % 2147483647) % 128
    elif row == 2:
        return ((item * p2 + 31 * 2) % 2147483647) % 128
    else:
        return ((item * p3 + 31 * 3) % 2147483647) % 128

def update(state, item):
    import struct
    
    # state is a bytearray or bytes. We need to return bytes.
    # Convert to bytearray for mutation if necessary, but we can just copy.
    s = bytearray(state)
    
    if 0 <= item < 20:
        # Hot item: exact count in first 20 bytes (10 uint16s)
        offset = item * 2
        current = struct.unpack_from('<H', s, offset)[0]
        # Increment. Max count is 1024, fits in uint16.
        new_val = current + 1
        struct.pack_into('<H', s, offset, new_val)
    else:
        # Cold item: update CM sketch
        # CM sketch starts at byte 20
        base_offset = 20
        for row in range(4):
            idx = _cm_index(item, row)
            offset = base_offset + row * (128 * 2) + idx * 2
            current = struct.unpack_from('<H', s, offset)[0]
            new_val = current + 1
            struct.pack_into('<H', s, offset, new_val)
            
    return bytes(s)

def query(state, item):
    import struct
    
    s = state
    
    if 0 <= item < 20:
        # Hot item: return exact count
        offset = item * 2
        return struct.unpack_from('<H', s, offset)[0]
    else:
        # Cold item: return min of CM sketch
        base_offset = 20
        min_count = 1024  # Max possible count
        for row in range(4):
            idx = _cm_index(item, row)
            offset = base_offset + row * (128 * 2) + idx * 2
            val = struct.unpack_from('<H', s, offset)[0]
            if val < min_count:
                min_count = val
        return min_count

# EVOLVE-BLOCK-END
