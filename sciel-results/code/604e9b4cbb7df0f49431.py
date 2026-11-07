def update(state, item):
    buf = bytearray(state)
    idx = item
    val = buf[idx]
    if val < 255:
        val += 1
    buf[idx] = val
    return bytes(buf)
