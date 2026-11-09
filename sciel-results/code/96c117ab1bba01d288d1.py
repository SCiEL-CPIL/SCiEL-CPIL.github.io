import struct
import random

S = 510

def update(state, item):
    # Unpack
    total = struct.unpack_from('<I', state, 0)[0]
    # Items are at offset 4
    # Copy state to bytearray
    buf = bytearray(state)
    
    total += 1
    struct.pack_into('<I', buf, 0, total)
    
    if total <= S:
        # Add to reservoir
        offset = 4 + (total - 1) * 2
        struct.pack_into('<H', buf, offset, item)
    else:
        # Reservoir sampling
        # Generate random j in [0, total-1]
        # We need a deterministic random? No, random is fine, but state must persist.
        # The random seed doesn't persist, so we use the system random.
        # This is allowed? "Each 256-item chunk runs in a fresh interpreter".
        # So random state resets. This is fine, we just need the reservoir to be correct.
        j = random.randint(0, total - 1)
        if j < S:
            offset = 4 + j * 2
            struct.pack_into('<H', buf, offset, item)
            
    return bytes(buf)

def query(state, item):
    total = struct.unpack_from('<I', state, 0)[0]
    if total == 0:
        return 0
        
    count = 0
    # Iterate through reservoir
    # Only iterate up to min(total, S)
    n_items = min(total, S)
    for i in range(n_items):
        offset = 4 + i * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val == item:
            count += 1
            
    if total <= S:
        return float(count)
    else:
        return float(count) * total / S
