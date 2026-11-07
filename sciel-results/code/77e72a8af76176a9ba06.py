import struct

def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    State is a bytes object of length 1024.
    We store 400 uint16 counters in the first 800 bytes.
    """
    buf = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', buf, offset)[0]
    val += 1
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    """
    Query the frequency estimate for the given item.
    Returns the exact count stored in the state.
    """
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
