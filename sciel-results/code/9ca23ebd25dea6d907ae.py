def update(state, item):
    # state is bytes, len 1024
    # item is int 0..399
    new_state = bytearray(state)
    offset = item * 2
    # Read current count
    # Little endian uint16
    current = new_state[offset] | (new_state[offset+1] << 8)
    new_count = current + 1
    # Write back
    new_state[offset] = new_count & 0xFF
    new_state[offset+1] = (new_count >> 8) & 0xFF
    return bytes(new_state)

def query(state, item):
    offset = item * 2
    return state[offset] | (state[offset+1] << 8)
