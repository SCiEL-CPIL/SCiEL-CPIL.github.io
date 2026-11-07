    def update(state, item):
        b = bytearray(state)
        # item is 0..399
        # We have 512 slots of uint16.
        # Map item to slot item.
        idx = item
        offset = idx * 2
        val = int.from_bytes(b[offset:offset+2], 'little')
        val += 1
        b[offset:offset+2] = val.to_bytes(2, 'little')
        return bytes(b)

    def query(state, item):
        idx = item
        offset = idx * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        return val
    