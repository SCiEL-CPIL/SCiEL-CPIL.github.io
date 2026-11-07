import struct

# 400 items * 2 bytes (uint16) = 800 bytes.
# The rest of the 1024 bytes are unused (zeros).
ITEMS = 400

def update(state, item):
    buf = bytearray(state)
    offset = item * 2
    # Read current count
    val = struct.unpack_from('<H', buf, offset)[0]
    # Increment
    val += 1
    # Write back
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
