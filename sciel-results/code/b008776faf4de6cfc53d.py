"""
Dynamic Heavy Hitter + Count-Min Sketch:
- State layout:
  - Bytes 0-199: 100 candidate slots, each storing (item_id: 1 byte, count: 2 bytes uint16).
  - Bytes 200-1023: 824 bytes for a Count-Min Sketch (2 rows of 206 uint16 counters).
- update:
  - Find candidate slot for item. If found, increment count.
  - If not found, find a slot with count 0 (empty). If none, find slot with min count.
  - If min count is 0, assign new item. Else, increment min count (collision) and update HH to that min item.
  - Always increment CM sketch counters for the item.
- query:
  - Check HH slots. If item found, return max(slot_count, cm_estimate).
  - Else, return cm_estimate.
"""

import struct

_HH_SLOTS = 100
_HH_SLOT_SIZE = 3  # 1 byte item, 2 bytes count
_HH_BYTES = _HH_SLOTS * _HH_SLOT_SIZE  # 300 bytes

_CM_OFFSET = _HH_BYTES
_CM_DEPTH = 2
_CM_WIDTH = 206  # 2 * 206 * 2 = 824 bytes. 300 + 824 = 1124? No.
# Let's recalculate size.
# Total 1024 bytes.
# CM sketch should be as large as possible.
# If HH uses 300 bytes, CM has 724 bytes.
# 724 / 2 (bytes per counter) = 362 counters total.
# 2 rows -> 181 counters per row.
_CM_WIDTH = 181
_CM_BYTES = _CM_DEPTH * _CM_WIDTH * 2  # 2 * 181 * 2 = 724 bytes
# Total: 300 + 724 = 1024. Correct.

# Hash parameters for CM sketch
_A = [1035493289, 775374203]
_B = [0, 129]

def _read_uint16(buf, offset):
    return buf[offset] | (buf[offset + 1] << 8)

def _write_uint16(buf, offset, val):
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    # 1. Update CM Sketch
    for i in range(_CM_DEPTH):
        idx = (_A[i] * item + _B[i]) % _CM_WIDTH
        byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
        val = _read_uint16(buf, byte_offset)
        val += 1
        _write_uint16(buf, byte_offset, val)
        
    # 2. Update HH Table
    # Find slot for item
    found_slot = -1
    min_count = 65535
    min_slot = -1
    
    for s in range(_HH_SLOTS):
        off = s * _HH_SLOT_SIZE
        slot_item = buf[off]
        slot_count = _read_uint16(buf, off + 1)
        
        if slot_item == item:
            found_slot = s
            break
            
        if slot_count < min_count:
            min_count = slot_count
            min_slot = s
            
    if found_slot != -1:
        # Increment existing slot
        off = found_slot * _HH_SLOT_SIZE
        val = _read_uint16(buf, off + 1)
        val += 1
        _write_uint16(buf, off + 1, val)
    else:
        # Check if min_slot is empty (count 0)
        # Note: Initial state is zeros, so item=0, count=0 means empty.
        # But if item=0 is a real item, we need to distinguish.
        # Since items are 0..399, and initial state is 0 bytes, 
        # a slot (0, 0) is ambiguous.
        # However, if item=0 appears, we should track it.
        # If min_count == 0, it means the slot is "empty" OR contains item 0 with count 0 (impossible after first update to 0).
        # Wait, if item 0 is inserted, count becomes 1. So (0,0) is strictly empty.
        
        if min_count == 0:
            # Assign new item to empty slot
            off = min_slot * _HH_SLOT_SIZE
            buf[off] = item
            _write_uint16(buf, off + 1, 1)
        else:
            # Collision: increment min slot's count and change its item to current item
            # This is a standard "Space Saving" or "Heavy Hitter" eviction strategy.
            off = min_slot * _HH_SLOT_SIZE
            # buf[off] = item  # We update the item identity to the new one
            # val = min_count + 1
            # _write_uint16(buf, off + 1, val)
            
            # Actually, standard HH algorithms often just increment the count and keep the old item?
            # No, that loses the new item's count.
            # "Space Saving" algorithm: 
            #   If item not found, find min counter. 
            #   Increment min counter. 
            #   If the new count is higher than the old min, we might swap? 
            #   Actually, Space Saving says: increment the min counter, and if the item was not present, set the item to the new one.
            #   But wait, if we set the item to the new one, we lose the count of the old item.
            #   The error introduced is bounded by the number of evictions.
            
            # Let's use the Space Saving logic:
            # 1. Find min slot.
            # 2. Increment its count.
            # 3. Set its item to the current item.
            # This ensures the count of the current item is at least 1 (in that slot).
            # The error for any item is bounded by the number of times it was evicted.
            
            buf[off] = item
            _write_uint16(buf, off + 1, min_count + 1)
            
    return bytes(buf)

def query(state, item):
    buf = state
    
    # 1. Get CM Estimate
    cm_min = 65535
    for i in range(_CM_DEPTH):
        idx = (_A[i] * item + _B[i]) % _CM_WIDTH
        byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
        val = _read_uint16(buf, byte_offset)
        if val < cm_min:
            cm_min = val
            
    # 2. Check HH Table
    hh_val = 0
    for s in range(_HH_SLOTS):
        off = s * _HH_SLOT_SIZE
        slot_item = buf[off]
        if slot_item == item:
            slot_count = _read_uint16(buf, off + 1)
            if slot_count > hh_val:
                hh_val = slot_count
            # We can break if we find it? 
            # In Space Saving, an item should only be in one slot.
            # But due to collisions/evictions, could it be in multiple?
            # No, we only assign to a slot if not found. 
            # However, if item A is evicted by B, A is gone.
            # If A reappears, it might find a different empty slot or evict another.
            # So it's possible for an item to appear in multiple slots? 
            # No, because if it's in slot 1, we find it and increment. We don't create a new entry.
            # So it should be unique.
            break
            
    # Return max of HH and CM
    # HH count is a lower bound (underestimate) for the true count?
    # Actually, HH count can be an overestimate if collisions occurred (count incremented for other items).
    # CM count is an overestimate due to collisions.
    # True count is between them? Not necessarily.
    # But taking the max is a common heuristic.
    # If HH is 0, we rely on CM.
    # If HH is > 0, it's likely a heavy hitter.
    
    return max(hh_val, cm_min)

