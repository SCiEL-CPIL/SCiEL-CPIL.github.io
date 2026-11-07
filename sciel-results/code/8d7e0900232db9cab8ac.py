import struct
import time

# Exact frequency storage: 400 items, each count fits in 1 byte (max 1024 < 256? No, 1024 > 255)
# Wait, max count is 1024. 1 byte is not enough. We need 2 bytes per item.
# 400 items * 2 bytes = 800 bytes.
# Total state is 1024 bytes.
# Remaining 224 bytes can be used for padding or other purposes.
# Since we only query items 0..399, we can just store the exact counts.

def update(state, item):
    # state is bytes of length 1024
    # We store counts for items 0..399 in the first 800 bytes (400 * 2 bytes)
    # Each count is a little-endian uint16.
    
    # Convert to bytearray for mutation
    buf = bytearray(state)
    
    # Item is in 0..399
    if 0 <= item < 400:
        offset = item * 2
        # Read current count
        val = struct.unpack_from('<H', buf, offset)[0]
        # Increment
        val += 1
        # Write back
        struct.pack_into('<H', buf, offset, val)
    
    return bytes(buf)

def query(state, item):
    # state is bytes of length 1024
    # Return the exact count for item
    if 0 <= item < 400:
        offset = item * 2
        val = struct.unpack_from('<H', state, offset)[0]
        return val
    else:
        return 0
