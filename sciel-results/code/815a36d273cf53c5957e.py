import struct

# Primes for hashing
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647
CM_ROWS = 4
CM_COLS = 120
HH_SIZE = 16

def _get_indices(item):
    indices = []
    for row in range(CM_ROWS):
        idx = ((item * PRIMES[row] + 31 * row) % MOD) % CM_COLS
        indices.append(idx)
    return indices

def update(state, item):
    # Parse state
    # First 64 bytes: 16 slots of (2 byte ID, 2 byte Count)
    # Next 960 bytes: 4 * 120 uint16 counters
    
    hh_slots = []
    for i in range(HH_SIZE):
        offset = i * 4
        item_id, count = struct.unpack_from('<HH', state, offset)
        hh_slots.append([item_id, count])
    
    # CM Sketch is in the rest of the bytes
    # We need to modify it in place or create a new one
    # It's easier to work with a bytearray
    
    new_state = bytearray(state)
    
    # Check if item is in HH
    in_hh = False
    hh_index = -1
    for i in range(HH_SIZE):
        if hh_slots[i][0] == item:
            in_hh = True
            hh_index = i
            break
            
    if in_hh:
        # Increment count
        new_count = hh_slots[hh_index][1] + 1
        hh_slots[hh_index][1] = new_count
        # Write back to state
        struct.pack_into('<HH', new_state, hh_index * 4, item, new_count)
    else:
        # Item not in HH
        # Check if HH is full
        is_full = all(slot[0] != 0 or slot[1] > 0 for slot in hh_slots) 
        # Note: Item 0 is valid. We need a better "empty" check.
        # Let's use count == 0 as empty marker?
        # But count can be 0 if decremented to 0.
        # Let's assume empty slot has count 0 and item 0 (or any).
        # Actually, if count is 0, it's effectively empty.
        
        empty_index = -1
        for i in range(HH_SIZE):
            if hh_slots[i][1] == 0:
                empty_index = i
                break
                
        if empty_index != -1:
            # Insert into empty slot
            hh_slots[empty_index] = [item, 1]
            struct.pack_into('<HH', new_state, empty_index * 4, item, 1)
        else:
            # HH is full
            # Find min count
            min_count = min(slot[1] for slot in hh_slots)
            
            if min_count == 1:
                # Replace one of the min count slots
                for i in range(HH_SIZE):
                    if hh_slots[i][1] == 1:
                        hh_slots[i] = [item, 1]
                        struct.pack_into('<HH', new_state, i * 4, item, 1)
                        break
            else:
                # Decrement all
                for i in range(HH_SIZE):
                    new_c = hh_slots[i][1] - 1
                    hh_slots[i][1] = new_c
                    struct.pack_into('<HH', new_state, i * 4, hh_slots[i][0], new_c)
                
                # Check if any became 0
                empty_index = -1
                for i in range(HH_SIZE):
                    if hh_slots[i][1] == 0:
                        empty_index = i
                        break
                        
                if empty_index != -1:
                    hh_slots[empty_index] = [item, 1]
                    struct.pack_into('<HH', new_state, empty_index * 4, item, 1)
                else:
                    # No space, ignore item for HH
                    pass

    # Update CM Sketch if item was NOT in HH before update
    if not in_hh:
        indices = _get_indices(item)
        for row in range(CM_ROWS):
            idx = indices[row]
            # Offset in state: 64 (HH) + row * 120 * 2 (bytes per row) + idx * 2
            offset = 64 + row * (CM_COLS * 2) + idx * 2
            current_val, = struct.unpack_from('<H', new_state, offset)
            new_val = current_val + 1
            # Cap at 65535
            if new_val > 65535:
                new_val = 65535
            struct.pack_into('<H', new_state, offset, new_val)

    return bytes(new_state)

def query(state, item):
    # Parse HH
    for i in range(HH_SIZE):
        offset = i * 4
        item_id, count = struct.unpack_from('<HH', state, offset)
        if item_id == item and count > 0:
            return float(count)
            
    # Not in HH, query CM
    indices = _get_indices(item)
    min_val = 65535
    for row in range(CM_ROWS):
        idx = indices[row]
        offset = 64 + row * (CM_COLS * 2) + idx * 2
        val, = struct.unpack_from('<H', state, offset)
        if val < min_val:
            min_val = val
            
    return float(min_val)
