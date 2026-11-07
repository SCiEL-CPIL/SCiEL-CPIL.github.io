def update(state, item):
    buf = bytearray(state)
    
    if item < 20:
        offset = item * 2
        # Read uint16 LE
        val = buf[offset] | (buf[offset + 1] << 8)
        val += 1
        # Write uint16 LE
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    else:
        offset = 40 + (item - 20)
        val = buf[offset]
        val += 1
        if val > 255:
            val = 255
        buf[offset] = val
        
    return bytes(buf)

def query(state, item):
    if item < 20:
        offset = item * 2
        return state[offset] | (state[offset + 1] << 8)
    else:
        offset = 40 + (item - 20)
        return state[offset]
