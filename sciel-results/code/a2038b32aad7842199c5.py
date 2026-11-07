import struct

def update(state, item):
    """
    Increment the exact count for 'item' in the state.
    State layout: 1024 bytes.
    Item i (0..399) is stored as a little-endian uint16 at offset 2*i.
    Total used: 800 bytes.
    """
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
    """
    Return the exact frequency estimate for 'item'.
    """
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
