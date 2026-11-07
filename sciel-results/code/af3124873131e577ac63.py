"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
def update(state, item):
    b = bytearray(state)
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 24) & 0xFF
    offset = idx * 4
    val = int.from_bytes(b[offset:offset+4], 'little')
    val = (val + 1) & 0xFFFFFFFF
    b[offset:offset+4] = val.to_bytes(4, 'little')
    return bytes(b)

def query(state, item):
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 24) & 0xFF
    offset = idx * 4
    val = int.from_bytes(state[offset:offset+4], 'little')
    return val
