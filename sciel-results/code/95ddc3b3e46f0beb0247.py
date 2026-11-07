def update(state, item):
    b = bytearray(state)
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 22) & 0x3FF  # 1024 counters
    b[idx] = (b[idx] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 22) & 0x3FF
    return state[idx]
