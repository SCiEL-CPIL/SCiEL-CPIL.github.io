import struct

def update(state, item):
    buf = bytearray(state)
    # Item is 0..399
    # Offset is item * 2
    idx = item * 2
    # Read uint16 LE
    val = buf[idx] | (buf[idx+1] << 8)
    # Increment
    val += 1
    # Write back
    buf[idx] = val & 0xFF
    buf[idx+1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    idx = item * 2
    val = state[idx] | (state[idx+1] << 8)
    return val
