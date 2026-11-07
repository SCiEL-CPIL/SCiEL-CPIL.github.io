"""
Improved frequency sketch: Hybrid 8-bit base counters + Top-16 high-resolution counters.
State Layout (1024 bytes):
- Bytes 0-399: 8-bit base counters for items 0-399.
- Bytes 400-463: Top-16 tracker. 16 entries of 4 bytes each.
  - Each entry: 2 bytes Item ID (uint16 LE), 2 bytes Delta (uint16 LE).
  - Delta = (True Count - Base Count) for that item, if it's in the top-16.
  - If an item is not in the top-16, its delta is 0 (conceptually).
- Bytes 464-1023: Unused/Zero.
"""

import struct

def _parse_top_k(state):
    """
    Parse the top-16 section from state.
    Returns a list of (item_id, delta) tuples.
    """
    top_k = []
    offset = 400
    for _ in range(16):
        item_id = struct.unpack_from('<H', state, offset)[0]
        delta = struct.unpack_from('<H', state, offset + 2)[0]
        top_k.append((item_id, delta))
        offset += 4
    return top_k

def _serialize_top_k(top_k):
    """
    Serialize the top-16 list back into bytes.
    """
    buf = bytearray(64)
    for i, (item_id, delta) in enumerate(top_k):
        struct.pack_into('<H', buf, i * 4, item_id)
        struct.pack_into('<H', buf, i * 4 + 2, delta)
    return bytes(buf)

def update(state, item):
    """
    Update the state by incrementing the count for 'item'.
    """
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(1024)
    state = bytearray(state)
    if len(state) < 1024:
        state.extend(b'\x00' * (1024 - len(state)))
    
    # 1. Update base counter
    if state[item] < 255:
        state[item] += 1
    
    current_base = state[item]
    
    # 2. Parse top-k
    top_k = _parse_top_k(bytes(state[400:464]))
    
    # 3. Check if item is in top-k
    found = False
    min_idx = -1
    min_total = float('inf')
    
    for i, (id, delta) in enumerate(top_k):
        # Calculate total count for this top-k item
        # Note: base count for id is state[id]
        total = state[id] + delta
        
        if id == item:
            # Increment delta
            top_k[i] = (id, delta + 1)
            found = True
            break
        
        # Track minimum for potential replacement
        if total < min_total:
            min_total = total
            min_idx = i
            
    if not found:
        # Item is not in top-k. Check if it should replace the min item.
        # Current total count for 'item' is current_base + 0 (delta is 0)
        if current_base > min_total:
            # Replace min_idx with item
            # Delta = current_base - state[item] (which is 0)
            # Wait, if we replace, the new delta is current_base - state[item]?
            # No, state[item] is already incremented.
            # The "true" count is current_base.
            # The base count is current_base.
            # So delta = 0?
            # This logic is flawed. If we replace, we are saying this item has a count 
            # that is better represented by base + delta.
            # If count < 256, delta is 0.
            # If count > 255, base is 255, delta is count - 255.
            
            # Let's re-evaluate.
            # If item is not in top-k, its effective count is just base_count.
            # If we add it to top-k, we should set delta such that base + delta = true_count.
            # True count is current_base (if < 256) or 255 + (current_base - 255) ?
            # Actually, if current_base == 255, it might have overflowed.
            # But we don't know the true count if it overflowed.
            # However, if it overflowed, it's likely a high-frequency item.
            
            # Let's just set delta = 0 for now if it enters top-k via replacement.
            # It will accumulate delta in subsequent updates.
            top_k[min_idx] = (item, 0)
            
    # 4. Serialize top-k back
    state[400:464] = _serialize_top_k(top_k)
    
    return bytes(state)

def query(state, item):
    """
    Query the estimated count for 'item'.
    """
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(1024)
    if len(state) < 400:
        return 0
    
    base = int(state[item])
    
    # Check top-k
    top_k = _parse_top_k(bytes(state[400:464]))
    
    for id, delta in top_k:
        if id == item:
            return base + delta
            
    return base

