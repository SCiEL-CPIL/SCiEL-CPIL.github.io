def update(state, item):
    b = bytearray(state)
    
    # Row 1: hash using p1
    idx1 = ((item * 1035493289 + 31) % 2147483647) % 256
    off1 = idx1 * 2
    val1 = int.from_bytes(b[off1:off1+2], 'little')
    b[off1:off1+2] = (val1 + 1).to_bytes(2, 'little')
    
    # Row 2: hash using p2
    idx2 = ((item * 775374203 + 31) % 2147483647) % 256
    off2 = 512 + idx2 * 2
    val2 = int.from_bytes(b[off2:off2+2], 'little')
    b[off2:off2+2] = (val2 + 1).to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    idx1 = ((item * 1035493289 + 31) % 2147483647) % 256
    off1 = idx1 * 2
    val1 = int.from_bytes(state[off1:off1+2], 'little')
    
    idx2 = ((item * 775374203 + 31) % 2147483647) % 256
    off2 = 512 + idx2 * 2
    val2 = int.from_bytes(state[off2:off2+2], 'little')
    
    return min(val1, val2)
