def update(state, item):
    b = bytearray(state)
    idx = item * 2
    count = b[idx] | (b[idx+1] << 8)
    count += 1
    b[idx] = count & 0xFF
    b[idx+1] = (count >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    idx = item * 2
    return state[idx] | (state[idx+1] << 8)
