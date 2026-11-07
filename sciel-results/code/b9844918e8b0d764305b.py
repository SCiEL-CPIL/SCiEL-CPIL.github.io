"""
Hybrid frequency sketch: exact counting for top-256 items + 5x128 CM sketch for tail.
State layout (1024 bytes):
  - 4 bytes: header (num_exact, version)
  - 256*3 = 768 bytes: 256 entries of (item: uint16, count: uint16)
  - 5*128*2 = 1280 bytes... wait, that's too much.

Let me recalculate:
  - 256 exact entries: item (2 bytes) + count (2 bytes) = 4 bytes each = 1024 bytes. Too much!

Let me redesign:
  - We have 1024 bytes total.
  - Option: 128 exact entries (2 bytes item + 2 bytes count = 4 bytes each) = 512 bytes
  - Remaining: 512 bytes for CM sketch: 4 rows x 64 columns x 2 bytes (uint16) = 512 bytes.
  
  Or: 192 exact entries (4 bytes each) = 768 bytes, remaining 256 bytes for CM: 4x16x4 bytes or 2x64x2 bytes.

Actually, let me think about what gives best error reduction.

The key insight from the knowledge: exact counting for heavy hitters eliminates overestimation for high-frequency items. The CM sketch handles the tail.

Let me use:
  - 128 exact slots: 128 * (2 bytes item + 2 bytes count) = 512 bytes
  - 512 bytes remaining for CM sketch: 4 rows x 64 columns x 2 bytes (uint16) = 512 bytes

For update:
  - Check if item is in exact table. If yes, increment count.
  - If no, check if there's an empty slot. If yes, insert.
  - If table is full, use CM sketch only.
  - Always update CM sketch.

For query:
  - If item is in exact table, return exact count.
  - Otherwise, return CM sketch estimate.

This should significantly reduce error for heavy hitters (Zipf stream, stream 3 first half) while maintaining reasonable estimates for the tail.

Let me also consider: the CM sketch with uint16 counters can overflow if counts exceed 65535. With 1024 items total, max count is 1024, so uint16 is fine.

Actually, wait - the state is shared across all 3 streams (1024 items each = 3072 total items). So max count per item could be up to 3072 in the worst case, but realistically for Zipf(1.6) on 400 items, the top item might get ~100-200 occurrences per 1024-item stream, so ~300-600 total. uint16 max is 65535, so we're safe.

Let me implement this.
"""

# EVOLVE-BLOCK-START
import struct
import time

# Layout:
# Bytes 0-1: num_exact (uint16, number of valid exact entries)
# Bytes 2-3: padding/version
# Bytes 4 through 4+128*4-1 = 4 to 515: 128 exact entries, each 4 bytes (item: uint16 LE, count: uint16 LE)
# Bytes 516 through 516+4*64*2-1 = 516 to 1023: 4 rows x 64 cols x 2 bytes (uint16 LE)

_NUM_EXACT = 128
_EXACT_SIZE = 4  # bytes per entry
_EXACT_OFFSET = 4
_CM_ROWS = 4
_CM_COLS = 64
_CM_OFFSET = _EXACT_OFFSET + _NUM_EXACT * _EXACT_SIZE  # 4 + 512 = 516
_CM_SIZE = _CM_ROWS * _CM_COLS * 2  # 4*64*2 = 512

_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
]
_MOD = 2147483647  # 2^31 - 1
_MAX_U16 = 0xFFFF


def _find_exact_entry(buf, item):
    """Find exact entry index for item, or -1 if not found."""
    for i in range(_NUM_EXACT):
        offset = _EXACT_OFFSET + i * _EXACT_SIZE
        stored_item = buf[offset] | (buf[offset + 1] << 8)
        if stored_item == item:
            return i
    return -1


def _find_empty_exact_slot(buf, num_exact):
    """Find first empty exact slot, or -1 if full."""
    for i in range(num_exact):
        offset = _EXACT_OFFSET + i * _EXACT_SIZE
        stored_item = buf[offset] | (buf[offset + 1] << 8)
        stored_count = buf[offset + 2] | (buf[offset + 3] << 8)
        if stored_count == 0:
            return i
    return -1


def _cm_update(buf, item):
    """Update CM sketch in buf."""
    for row in range(_CM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _CM_COLS
        offset = _CM_OFFSET + (row * _CM_COLS + idx) * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        if val < _MAX_U16:
            val += 1
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF


def _cm_query(buf, item):
    """Query CM sketch from buf."""
    min_val = _MAX_U16
    for row in range(_CM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _CM_COLS
        offset = _CM_OFFSET + (row * _CM_COLS + idx) * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        if val < min_val:
            min_val = val
    return min_val


def update(state, item):
    buf = bytearray(state)
    
    # Read num_exact
    num_exact = buf[0] | (buf[1] << 8)
    
    # Check if item is in exact table
    entry_idx = _find_exact_entry(buf, item)
    
    if entry_idx >= 0:
        # Increment exact count
        offset = _EXACT_OFFSET + entry_idx * _EXACT_SIZE
        count = buf[offset + 2] | (buf[offset + 3] << 8)
        if count < _MAX_U16:
            count += 1
        buf[offset + 2] = count & 0xFF
        buf[offset + 3] = (count >> 8) & 0xFF
    else:
        # Try to find empty slot
        empty_idx = _find_empty_exact_slot(buf, num_exact)
        if empty_idx >= 0:
            # Insert new entry
            offset = _EXACT_OFFSET + empty_idx * _EXACT_SIZE
            buf[offset] = item & 0xFF
            buf[offset + 1] = (item >> 8) & 0xFF
            buf[offset + 2] = 1  # count = 1
            buf[offset + 3] = 0
            if empty_idx >= num_exact:
                num_exact = empty_idx + 1
                buf[0] = num_exact & 0xFF
                buf[1] = (num_exact >> 8) & 0xFF
    
    # Always update CM sketch
    _cm_update(buf, item)
    
    return bytes(buf)


def query(state, item):
    buf = state
    
    # Read num_exact
    num_exact = buf[0] | (buf[1] << 8)
    
    # Check if item is in exact table
    entry_idx = _find_exact_entry(buf, item)
    
    if entry_idx >= 0:
        offset = _EXACT_OFFSET + entry_idx * _EXACT_SIZE
        count = buf[offset + 2] | (buf[offset + 3] << 8)
        return count
    
    # Otherwise, use CM sketch
    return _cm_query(buf, item)

# EVOLVE-BLOCK-END
