import struct
import sys

# 400 items, 2 bytes each = 800 bytes for direct counters
# Remaining 224 bytes: 7 rows x 8 cols x 4 bytes = 224 bytes for fallback CM sketch
NUM_ITEMS = 400
DIRECT_BYTES = NUM_ITEMS * 2  # 800 bytes
FALLBACK_ROWS = 7
FALLBACK_COLS = 8
FALLBACK_BYTES = FALLBACK_ROWS * FALLBACK_COLS * 4  # 224 bytes
STATE_SIZE = DIRECT_BYTES + FALLBACK_BYTES  # 1024 bytes

# Primes for fallback sketch
FALLBACK_PRIMES = (1035493289, 775374203, 1495862053, 2015145641,
                   374761393, 641194019, 1274126177)
FALLBACK_MOD = 2147483647


def _fallback_index(item, row):
    p = FALLBACK_PRIMES[row]
    return ((item * p + 31 * row) % FALLBACK_MOD) % FALLBACK_COLS


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    if not isinstance(state, bytes) or len(state) != STATE_SIZE:
        # Try to handle gracefully
        state = bytes(STATE_SIZE)
    
    item = int(item)
    if item < 0 or item >= NUM_ITEMS:
        return state  # Invalid item, return unchanged
    
    # Parse state
    # Direct counters: first 800 bytes, 400 uint16 little-endian
    # Fallback: next 224 bytes, 7*8 uint32 little-endian
    
    # Use bytearray for in-place modification
    buf = bytearray(state)
    
    # Update direct counter
    off = item * 2
    val = struct.unpack_from('<H', buf, off)[0]
    if val < 65535:
        val += 1
    # If val == 65535, we rely on fallback sketch
    
    struct.pack_into('<H', buf, off, val)
    
    # Update fallback sketch (always, as backup)
    base_off = DIRECT_BYTES
    for row in range(FALLBACK_ROWS):
        col = _fallback_index(item, row)
        idx = base_off + (row * FALLBACK_COLS + col) * 4
        cur = struct.unpack_from('<I', buf, idx)[0]
        if cur < 4294967295:
            cur += 1
        struct.pack_into('<I', buf, idx, cur)
    
    return bytes(buf)


def query(state, item):
    """Query the frequency estimate for an item. Returns nonnegative float."""
    if not isinstance(state, bytes) or len(state) != STATE_SIZE:
        return 0.0
    
    item = int(item)
    if item < 0 or item >= NUM_ITEMS:
        return 0.0
    
    # Get direct counter
    off = item * 2
    direct_val = struct.unpack_from('<H', state, off)[0]
    
    # If direct counter hasn't overflowed, it's exact
    if direct_val < 65535:
        return float(direct_val)
    
    # Fallback: use CM sketch min
    base_off = DIRECT_BYTES
    min_val = float('inf')
    for row in range(FALLBACK_ROWS):
        col = _fallback_index(item, row)
        idx = base_off + (row * FALLBACK_COLS + col) * 4
        val = struct.unpack_from('<I', state, idx)[0]
        if val < min_val:
            min_val = val
    
    # Return the larger of direct and fallback (direct is at least the true count
    # minus overflow, fallback is an upper bound)
    # Actually, direct_val == 65535 means we lost count. Fallback is an upper bound.
    # Best estimate: max(direct_val, min_val) but min_val >= true count.
    # Since direct_val = 65535 < true count (in overflow case), use min_val.
    return float(min_val)
