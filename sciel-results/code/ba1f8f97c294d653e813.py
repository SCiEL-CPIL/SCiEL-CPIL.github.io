"""Domain-adaptive frequency sketch: hot exact counters + widened CM sketch."""
import struct
import time

# EVOLVE-BLOCK-START
# Precompute hash indices for the CM sketch part (items 20..399)
# We use 8 rows and 32 columns for the CM sketch.
# State layout:
# Bytes 0..19: uint32 LE counters for items 0..19 (exact)
# Bytes 20..20+8*32*4-1: CM sketch counters (8 rows * 32 cols * 4 bytes = 1024 bytes)
# Total: 20 + 1024 = 1044 bytes. Wait, limit is 1024 bytes.
# Let's adjust.
# If we use 20 bytes for hot items, we have 1004 bytes left.
# 1004 / 4 = 251 counters.
# Let's use 7 rows of 32 counters = 224 counters = 896 bytes.
# Total: 20 + 896 = 916 bytes.
# Or 8 rows of 31 counters = 248 counters = 992 bytes.
# Total: 20 + 992 = 1012 bytes. This fits.
# Let's stick to powers of 2 or simple numbers for modulo? 
# Using modulo 31 is fine.

# Actually, let's reconsider the state size.
# 1024 bytes total.
# Hot items: 20 items. Let's store them as uint32. 20 * 4 = 80 bytes.
# Remaining: 1024 - 80 = 944 bytes.
# 944 / 4 = 236 counters.
# Let's use 7 rows of 33 counters = 231 counters = 924 bytes.
# Total: 80 + 924 = 1004 bytes.
# Or 8 rows of 29 counters = 232 counters = 928 bytes.
# Total: 80 + 928 = 1008 bytes.

# Let's use a simpler approach:
# 20 hot items stored as uint32 (80 bytes).
# 8 rows, 30 cols CM sketch. 8 * 30 * 4 = 960 bytes.
# Total: 80 + 960 = 1040 bytes. Too big.

# 20 hot items stored as uint32 (80 bytes).
# 7 rows, 32 cols CM sketch. 7 * 32 * 4 = 896 bytes.
# Total: 80 + 896 = 976 bytes. This is safe.

HOT_ITEMS = 20
CM_ROWS = 7
CM_COLS = 32
CM_START_OFFSET = HOT_ITEMS * 4  # 80

# Hash constants for CM sketch
# We need 7 different primes/constants.
P = [1035493289, 775374203, 1495862053, 2015145641, 1234567891, 987654321, 555555555]
M = 2147483647

def _cm_index(item, row):
    # Only called for item >= HOT_ITEMS
    return ((item * P[row] + 31 * row) % M) % CM_COLS

def update(state, item):
    if not isinstance(state, bytes) or len(state) != 1024:
        # Fallback or error handling? Contract says initial is 1024 zeros.
        # If state is invalid, we should probably return a valid state.
        # But the contract says "update returning non-bytes or over 1024 bytes" is invalid.
        # So we must return exactly 1024 bytes.
        # If input is bad, we can't fix it easily, but let's assume input is valid.
        pass
        
    # Convert to bytearray for mutation
    state = bytearray(state)
    
    if item < HOT_ITEMS:
        # Exact counter
        offset = item * 4
        val = struct.unpack_from('<I', state, offset)[0]
        val += 1
        struct.pack_into('<I', state, offset, val)
    else:
        # CM Sketch update
        for row in range(CM_ROWS):
            idx = _cm_index(item, row)
            offset = CM_START_OFFSET + (row * CM_COLS + idx) * 4
            val = struct.unpack_from('<I', state, offset)[0]
            val += 1
            struct.pack_into('<I', state, offset, val)
            
    return bytes(state)

def query(state, item):
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0
        
    if item < HOT_ITEMS:
        # Exact counter
        offset = item * 4
        val = struct.unpack_from('<I', state, offset)[0]
        return float(val)
    else:
        # CM Sketch query
        min_val = float('inf')
        for row in range(CM_ROWS):
            idx = _cm_index(item, row)
            offset = CM_START_OFFSET + (row * CM_COLS + idx) * 4
            val = struct.unpack_from('<I', state, offset)[0]
            if val < min_val:
                min_val = val
        return float(min_val)
# EVOLVE-BLOCK-END
