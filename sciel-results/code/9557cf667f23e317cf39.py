"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

# Precompute hash indices for all 400 items across 4 rows
# Layout: 400 bytes for exact counts (item 0..399 at index 0..399)
#         624 bytes for CMS: 4 rows x 156 columns (uint8)
# Total: 400 + 624 = 1024 bytes

def _precompute_indices():
    """Precompute CMS indices for all 400 items."""
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    indices = []
    for item in range(400):
        row_indices = []
        for row in range(4):
            p = p_values[row]
            idx = ((item * p + 31 * row) % 2147483647) % 156
            row_indices.append(idx)
        indices.append(row_indices)
    return indices

_CMS_INDICES = _precompute_indices()

def update(state, item):
    b = bytearray(state)
    
    # Update exact count for this item (byte 0-399)
    exact_count = b[item] + 1
    b[item] = min(exact_count, 255)
    
    # Update CMS counters (bytes 400 onwards)
    # 4 rows x 156 columns
    for row in range(4):
        col = _CMS_INDICES[item][row]
        offset = 400 + row * 156 + col
        b[offset] = (b[offset] + 1) & 0xFF
    
    return bytes(b)

def query(state, item):
    # Get exact count
    exact_count = state[item]
    
    # Get CMS estimate (min across 4 rows)
    cms_min = 255
    for row in range(4):
        col = _CMS_INDICES[item][row]
        offset = 400 + row * 156 + col
        val = state[offset]
        if val < cms_min:
            cms_min = val
    
    # Return the minimum of exact and CMS to be conservative
    # Exact count is reliable if < 255 (no overflow)
    # CMS may overestimate due to collisions
    if exact_count < 255:
        # Exact count hasn't overflowed, trust it but also consider CMS
        # Use min to avoid CMS overestimation
        return min(exact_count, cms_min)
    else:
        # Exact count may have overflowed, use CMS
        return cms_min

# EVOLVE-BLOCK-END
