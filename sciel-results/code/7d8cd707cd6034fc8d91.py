def update(state, item):
    # state is 1024 bytes, first 800 bytes are 400 uint16 counters
    # item is 0..399
    # Increment the counter for this item
    import struct
    state = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    new_val = val + 1
    struct.pack_into('<H', state, offset, new_val)
    return bytes(state)

def query(state, item):
    import struct
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)
