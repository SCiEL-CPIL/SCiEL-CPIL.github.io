"""Memory-bounded streaming frequency sketch: CMS + Bloom filter."""

import struct

# CMS parameters
CMS_ROWS = 3
CMS_COLS = 64
CMS_BYTES = CMS_ROWS * CMS_COLS * 4  # 768 bytes

# Bloom filter parameters
BLOOM_BITS = 2048  # 256 bytes
BLOOM_HASHES = 3

# State layout: bytes 0..767 = CMS (3 rows * 64 cols * uint32 LE), bytes 768..1023 = Bloom filter
STATE_SIZE = 1024

# Precompute hash parameters for CMS
# Using the same hash structure as seed but with 3 rows instead of 4
_CMS_P = (1035493289, 775374203, 1495862053)  # 3 rows
_CMS_PRIME = 2147483647


def _bloom_set(state_list, item):
    """Set bits in Bloom filter for item."""
    for i in range(BLOOM_HASHES):
        # Simple hash functions
        h = (item * (1000003 + 97 * i) + 7919 * i + 31) % BLOOM_BITS
        byte_idx = h // 8
        bit_idx = h % 8
        state_list[byte_idx] |= (1 << bit_idx)


def _bloom_test(state, item):
    """Test if item is in Bloom filter. state is bytes."""
    for i in range(BLOOM_HASHES):
        h = (item * (1000003 + 97 * i) + 7919 * i + 31) % BLOOM_BITS
        byte_idx = 768 + h // 8  # offset by CMS bytes
        bit_idx = h % 8
        if not (state[byte_idx] & (1 << bit_idx)):
            return False
    return True


def _cms_index(item, row):
    """Compute CMS column index for item and row."""
    p = _CMS_P[row]
    return ((item * p + 31 * row) % _CMS_PRIME) % CMS_COLS


def update(state, item):
    """Update sketch state with a new item."""
    # Convert to list of ints for mutation
    state_list = list(state)
    
    # Update CMS counters
    for row in range(CMS_ROWS):
        col = _cms_index(item, row)
        offset = row * CMS_COLS * 4 + col * 4
        # Read uint32 LE
        val = struct.unpack_from('<I', state_list, offset)
        # Actually struct needs bytes, let's do it manually
        val = state_list[offset] | (state_list[offset+1] << 8) | (state_list[offset+2] << 16) | (state_list[offset+3] << 24)
        val += 1
        state_list[offset] = val & 0xFF
        state_list[offset+1] = (val >> 8) & 0xFF
        state_list[offset+2] = (val >> 16) & 0xFF
        state_list[offset+3] = (val >> 24) & 0xFF
    
    # Update Bloom filter
    _bloom_set(state_list, item)
    
    return bytes(state_list)


def query(state, item):
    """Query frequency estimate for item."""
    # First check Bloom filter - if not present, return 0
    if not _bloom_test(state, item):
        return 0.0
    
    # Get CMS estimate (minimum over rows)
    min_val = float('inf')
    for row in range(CMS_ROWS):
        col = _cms_index(item, row)
        offset = row * CMS_COLS * 4 + col * 4
        val = state[offset] | (state[offset+1] << 8) | (state[offset+2] << 16) | (state[offset+3] << 24)
        if val < min_val:
            min_val = val
    
    return float(min_val)
