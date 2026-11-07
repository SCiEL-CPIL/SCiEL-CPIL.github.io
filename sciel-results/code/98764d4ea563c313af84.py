import struct
import time

# EVOLVE-BLOCK-START

# We allocate 1024 bytes as follows:
# 1. 256 bytes for exact counts of items 0..255 (1 byte each, max 255)
# 2. 768 bytes for a Count-Min sketch: 12 rows x 16 counters, each 4 bytes (uint32)
#    Total CM bytes: 12 * 16 * 4 = 768
#    Total: 256 + 768 = 1024

# Constants for hashing in the CM sketch
# We use 12 distinct large odd constants (primes or near-primes)
# The first 4 are from the seed, we add 8 more
P_CONSTANTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    1618033989,
    1357246801,
    1111111111,
    999999937
]

MOD = 2147483647  # 2^31 - 1
CM_WIDTH = 16     # 16 counters per row
CM_ROWS = 12      # 12 rows
EXACT_ITEMS = 256 # Items 0..255 get exact counting

def update(state, item):
    s = bytearray(state)
    
    # If item is in 0..255, increment its exact counter
    if item < EXACT_ITEMS:
        s[item] = min(s[item] + 1, 255)
    else:
        # For items >= 256, update the CM sketch
        # CM sketch starts at byte offset 256
        cm_offset = EXACT_ITEMS
        
        for row in range(CM_ROWS):
            p = P_CONSTANTS[row]
            # Hash function: ((item * p + row * 31) % MOD) % CM_WIDTH
            idx = ((item * p + row * 31) % MOD) % CM_WIDTH
            pos = cm_offset + row * CM_WIDTH * 4 + idx * 4
            
            # Read 4-byte little-endian uint32
            val = struct.unpack_from('<I', s, pos)[0]
            val += 1
            # Write back
            struct.pack_into('<I', s, pos, val)
    
    return bytes(s)

def query(state, item):
    if item < EXACT_ITEMS:
        # Return exact count
        return state[item]
    else:
        # Query the CM sketch: take minimum over all rows
        cm_offset = EXACT_ITEMS
        min_val = 1024  # Max possible count
        
        for row in range(CM_ROWS):
            p = P_CONSTANTS[row]
            idx = ((item * p + row * 31) % MOD) % CM_WIDTH
            pos = cm_offset + row * CM_WIDTH * 4 + idx * 4
            val = struct.unpack_from('<I', state, pos)[0]
            if val < min_val:
                min_val = val
        
        return min_val

# EVOLVE-BLOCK-END
