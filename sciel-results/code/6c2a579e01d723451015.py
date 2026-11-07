def update(state, item):
    b = bytearray(state)
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 23) & 0x1FF  # 512 counters
    offset = idx * 2
    val = int.from_bytes(b[offset:offset+2], 'little')
    val = (val + 1) & 0xFFFF
    b[offset:offset+2] = val.to_bytes(2, 'little')
    return bytes(b)

def query(state, item):
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 23) & 0x1FF
    offset = idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    return val
