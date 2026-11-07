import struct

# State layout: 1024 bytes
# 256 counters (uint16 each) = 512 bytes -> for heavy hitters
# 160 pairs of (item uint16, count uint16) = 640 bytes... too much
# Let's do: 200 counters (uint16) = 400 bytes + 312 bytes = 156 pairs? No.

# Better layout:
# Row 0: 256 uint16 counters for "exact" tracking of frequent items
# We need item identification. Let's use a different approach.

# Approach: Use 1024 bytes as:
# - 256 uint16 counters (512 bytes) for a small Count-Min-like structure won't fit item ids
# 
# Alternative: Store up to 256 (item, count) pairs using variable encoding is complex.
# Let's use fixed: 256 entries, each 4 bytes: item (uint16) + count (uint16) = 1024 bytes
# This is a "frequency table" with eviction of least-frequent item.

def update(state, item):
    # Unpack 256 pairs
    data = struct.unpack('<256I', state)
    data = list(data)
    
    found = -1
    min_idx = 0
    min_val = 0xFFFFFFFF
    
    for i in range(256):
        val = data[i]
        count = val & 0xFFFF
        if count < min_val:
            min_val = count
            min_idx = i
        # Check if item matches (item is in upper 16 bits)
        if (val >> 16) == item:
            found = i
            break
    
    if found >= 0:
        data[found] = (data[found] & 0xFFFF0000) | ((data[found] & 0xFFFF) + 1)
    else:
        # Evict the min count item
        data[min_idx] = (item << 16) | (min_val + 1)
    
    return struct.pack('<256I', *data)

def query(state, item):
    data = struct.unpack('<256I', state)
    for i in range(256):
        val = data[i]
        if (val >> 16) == item:
            return float(val & 0xFFFF)
    return 0.0
