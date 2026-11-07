"""Memory-bounded streaming frequency sketch with CMS + bitset."""
import struct
import time

# Layout: 50 bytes bitset (400 bits) + 4*121*4 bytes CMS = 50 + 1936 = 1986... 
# Wait, 4*121*4 = 1936, plus 50 = 1986 > 1024. Need to fit in 1024.
# 1024 - 50 = 974 bytes for CMS. 974 // (4*4) = 60.875 -> 60 columns.
# So 4 rows * 60 cols * 4 bytes = 960 bytes + 50 = 1010 <= 1024.
# Let's use 60 columns.

CMS_ROWS = 4
CMS_COLS = 60
CMS_BYTES = CMS_ROWS * CMS_COLS * 4  # 960
BITSET_BYTES = 50
TOTAL_BYTES = BITSET_BYTES + CMS_BYTES  # 1010

# Precompute hash parameters
P = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647

# Precompute hash indices for all items 0..399 for speed
_HASH_CACHE = {}

def _get_hash(item, row):
    key = (item, row)
    if key not in _HASH_CACHE:
        idx = ((item * P[row] + 31 * row) % MOD) % CMS_COLS
        _HASH_CACHE[key] = idx
    return _HASH_CACHE[key]

def update(state, item):
    """Update the sketch state with a new item."""
    # Parse current state
    bitset_bytes = state[:BITSET_BYTES]
    cms_bytes = state[BITSET_BYTES:]
    
    # Update bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    new_bitset = bytearray(bitset_bytes)
    new_bitset[byte_idx] |= bit_mask
    
    # Update CMS counters
    cms = list(struct.unpack(f'<{CMS_ROWS * CMS_COLS}I', cms_bytes))
    for row in range(CMS_ROWS):
        idx = _get_hash(item, row)
        pos = row * CMS_COLS + idx
        cms[pos] += 1
    
    # Pack back
    new_cms_bytes = struct.pack(f'<{CMS_ROWS * CMS_COLS}I', *cms)
    return bytes(new_bitset) + new_cms_bytes

def query(state, item):
    """Query the frequency estimate for an item."""
    bitset_bytes = state[:BITSET_BYTES]
    cms_bytes = state[BITSET_BYTES:]
    
    # Check if item was seen
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    seen = (bitset_bytes[byte_idx] & bit_mask) != 0
    
    if not seen:
        return 0.0
    
    # Get CMS estimate
    cms = struct.unpack(f'<{CMS_ROWS * CMS_COLS}I', cms_bytes)
    estimates = []
    for row in range(CMS_ROWS):
        idx = _get_hash(item, row)
        pos = row * CMS_COLS + idx
        estimates.append(cms[pos])
    
    return float(min(estimates))
