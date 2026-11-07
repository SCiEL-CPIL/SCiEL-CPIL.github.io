def update(state, item):
    # state is bytes, item is int
    ba = bytearray(state)
    # item is 0..399
    # offset = item * 2
    offset = item * 2
    # Read 16-bit LE
    current_val = ba[offset] | (ba[offset + 1] << 8)
    # Increment
    new_val = current_val + 1
    # Write back
    ba[offset] = new_val & 0xFF
    ba[offset + 1] = (new_val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    # state is bytes, item is int
    offset = item * 2
    val = state[offset] | (state[offset + 1] << 8)
    return val
