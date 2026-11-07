"""Hybrid sketch: 400 exact 8-bit counters + 8 heavy-hitter slots (16-bit) + CM sketch for overflow."""
import struct
import time

# --- Constants ---
N_ITEMS = 400
CM_ROWS = 4
CM_COLS = 64
CM_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
CM_MOD = 2147483647
HH_SLOTS = 8
MAX_8BIT = 255

# State layout (1024 bytes total):
#   [0:400]     400 x uint8  exact counters for items 0..399
#   [400:416]   8 x uint16   heavy-hitter item IDs (0..399) or 0xFFFF if empty
#   [416:432]   8 x uint16   heavy-hitter exact counts (16-bit)
#   [432:1024]  4 x 64 x uint32  CM sketch counters (little-endian)
# Total: 400 + 16 + 16 + 1024 = 1456 > 1024. Need to fit in 1024.
#
# Revised layout to fit 1024 bytes:
#   [0:400]     400 x uint8  exact counters
#   [400:416]   8 x uint16   HH item IDs
#   [416:432]   8 x uint16   HH counts
#   [432:1024]  592 bytes for CM sketch. 592 / 4 = 148 uint32 slots.
#   That's not 4x64=256. Let's reduce CM to fit.
#   1024 - 400 - 16 - 16 = 592 bytes for CM. 592/4 = 148 counters.
#   We can use 4 rows x 37 cols = 148. Or 2 rows x 74 = 148.
#   Let's use 4 rows x 37 cols.
#
# Actually, let's reconsider. The CM sketch is only needed for items whose
# 8-bit counter saturates. For Zipf(1.6) on 400 items, the top items can have
# counts well over 255. Let's keep the CM sketch but with fewer columns.
#
# Final layout:
#   [0:400]   400 uint8 exact counters
#   [400:416] 8 uint16 HH item IDs (0xFFFF = empty)
#   [416:432] 8 uint16 HH exact counts
#   [432:1024] 592 bytes = 148 uint32 CM counters, organized as 4 rows x 37 cols

CM_COLS_ACTUAL = 37
CM_TOTAL = CM_ROWS * CM_COLS_ACTUAL  # 148

def _cm_index(item, row):
    p = CM_PRIMES[row]
    return ((item * p + 31 * row) % CM_MOD) % CM_COLS_ACTUAL

def _cm_update(state, item):
    """Update CM sketch in-place (mutates the bytearray)."""
    for row in range(CM_ROWS):
        idx = _cm_index(item, row)
        pos = 432 + (row * CM_COLS_ACTUAL + idx) * 4
        val = struct.unpack_from('<I', state, pos)[0]
        val += 1
        struct.pack_into('<I', state, pos, val)

def _cm_query(state, item):
    """Query CM sketch, return min across rows."""
    mn = 0xFFFFFFFF
    for row in range(CM_ROWS):
        idx = _cm_index(item, row)
        pos = 432 + (row * CM_COLS_ACTUAL + idx) * 4
        val = struct.unpack_from('<I', state, pos)[0]
        if val < mn:
            mn = val
    return mn

def _hh_find_slot(state, item):
    """Find the HH slot index for item, or -1 if not present."""
    for i in range(HH_SLOTS):
        pos = 400 + i * 2
        hh_item = struct.unpack_from('<H', state, pos)[0]
        if hh_item == item:
            return i
    return -1

def _hh_min_slot(state):
    """Find the HH slot with the minimum count (for replacement)."""
    min_i = -1
    min_c = 0xFFFFFFFF
    for i in range(HH_SLOTS):
        pos = 416 + i * 2
        c = struct.unpack_from('<H', state, pos)[0]
        if c < min_c:
            min_c = c
            min_i = i
    return min_i, min_c

def update(state, item):
    """Update the state with a new item. Returns new state as bytes."""
    state = bytearray(state)
    
    # 1. Increment the 8-bit exact counter
    c8 = state[item]
    if c8 < MAX_8BIT:
        state[item] = c8 + 1
        c8 += 1
    else:
        # Counter is saturated; we rely on HH/CM for this item
        c8 = MAX_8BIT  # keep at max, HH will track exact
    
    # 2. Update CM sketch (always, for overflow correction)
    _cm_update(state, item)
    
    # 3. Update heavy-hitter table
    slot = _hh_find_slot(state, item)
    if slot != -1:
        # Already in HH table, increment its count
        pos = 416 + slot * 2
        hc = struct.unpack_from('<H', state, pos)[0]
        hc += 1
        if hc > 0xFFFF:
            hc = 0xFFFF
        struct.pack_into('<H', state, pos, hc)
    elif c8 > 0:  # only consider for HH if count is non-trivial
        # Check if we should add to HH table
        # Policy: add if c8 >= 16 (threshold to avoid filling with noise)
        # Actually, for correctness we want to track any item that might exceed 255.
        # But we only have 8 slots. Let's use a threshold.
        # For Zipf(1.6), items with count > 255 are rare. Let's set threshold = 100.
        if c8 >= 100:
            # Find a slot to use
            # First, try to find an empty slot (0xFFFF)
            empty = -1
            for i in range(HH_SLOTS):
                pos = 400 + i * 2
                hh_item = struct.unpack_from('<H', state, pos)[0]
                if hh_item == 0xFFFF:
                    empty = i
                    break
            if empty != -1:
                struct.pack_into('<H', state, 400 + empty * 2, item)
                struct.pack_into('<H', state, 416 + empty * 2, c8)
            else:
                # Replace the minimum-count HH entry if this item's count is larger
                min_i, min_c = _hh_min_slot(state)
                if c8 > min_c:
                    struct.pack_into('<H', state, 400 + min_i * 2, item)
                    struct.pack_into('<H', state, 416 + min_i * 2, c8)
    
    return bytes(state)

def query(state, item):
    """Query the estimated frequency of item. Returns a nonnegative float."""
    # 1. Check HH table first
    slot = _hh_find_slot(state, item)
    if slot != -1:
        pos = 416 + slot * 2
        hc = struct.unpack_from('<H', state, pos)[0]
        # HH count is exact (up to 16-bit limit)
        return float(hc)
    
    # 2. Use 8-bit counter
    c8 = state[item]
    if c8 < MAX_8BIT:
        # Not saturated, exact
        return float(c8)
    
    # 3. Saturated: use CM sketch for estimate
    cm_val = _cm_query(state, item)
    # CM overestimates; the true count is at least c8 (255) and at most cm_val
    # Best estimate: use cm_val, but it could be inflated by collisions
    # For safety, return cm_val (nonnegative, finite)
    return float(cm_val)
