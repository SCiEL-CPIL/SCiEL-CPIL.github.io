import struct

def update(state, item):
    buf = bytearray(state)
    byte_offset = item * 2
    val = struct.unpack_from('<H', buf, byte_offset)[0]
    val += 1
    struct.pack_into('<H', buf, byte_offset, val)
    return bytes(buf)

def query(state, item):
    byte_offset = item * 2
    val = struct.unpack_from('<H', state, byte_offset)[0]
    return val
