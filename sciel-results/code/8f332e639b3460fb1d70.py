import struct

# Primes from the seed program
P = [1035493289, 775374203, 1495862053, 2015145641]
M = 2147483647
C = 31

# Configuration
EXACT_COUNT = 32  # Number of items with exact counters
CMS_WIDTH = 112   # Number of counters per row in CMS
CMS_DEPTH = 4     # Number of rows in CMS

# Memory Layout:
# [0 : 128]       : 32 * uint32 (Exact counts for items 0-31)
# [128 : 128+224] : 112 * uint16 (CMS Row 0)
# [128+224 : ...] : 112 * uint16 (CMS Row 1)
# ...
# Total: 128 + 4 * 224 = 1024 bytes

def update(state, item):
    # Copy state to allow modification
    s = bytearray(state)
    
    if item < EXACT_COUNT:
        # Increment exact counter
        offset = item * 4
        val = struct.unpack_from('<I', s, offset)[0]
        val += 1
        struct.pack_into('<I', s, offset, val)
    else:
        # Update CMS
        base_offset = EXACT_COUNT * 4 # 128
        for r in range(CMS_DEPTH):
            # Hash function similar to seed
            idx = ((item * P[r] + C * r) % M) % CMS_WIDTH
            # Offset for this row and index
            # Each row is 112 * 2 = 224 bytes
            row_offset = base_offset + r * (CMS_WIDTH * 2)
            counter_offset = row_offset + idx * 2
            
            val = struct.unpack_from('<H', s, counter_offset)[0]
            val += 1
            struct.pack_into('<H', s, counter_offset, val)
            
    return bytes(s)

def query(state, item):
    if item < EXACT_COUNT:
        offset = item * 4
        val = struct.unpack_from('<I', state, offset)[0]
        return val
    else:
        base_offset = EXACT_COUNT * 4 # 128
        min_val = float('inf')
        for r in range(CMS_DEPTH):
            idx = ((item * P[r] + C * r) % M) % CMS_WIDTH
            row_offset = base_offset + r * (CMS_WIDTH * 2)
            counter_offset = row_offset + idx * 2
            val = struct.unpack_from('<H', state, counter_offset)[0]
            if val < min_val:
                min_val = val
        return min_val
