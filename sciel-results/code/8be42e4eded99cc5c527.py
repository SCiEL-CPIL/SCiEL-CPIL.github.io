import struct

def update(state, item):
    # Use the first 800 bytes to store exact counts for 400 items (uint16).
    # The remaining 224 bytes are preserved as padding.
    buf = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', buf, offset)[0]
    val += 1
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)
