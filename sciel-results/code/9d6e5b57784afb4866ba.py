import time

_START = time.time()
_TIMEOUT = 115

_NUM_ITEMS = 400

def update(state, item):
    if time.time() - _START > _TIMEOUT:
        return state
    
    buf = bytearray(state)
    
    if 0 <= item < _NUM_ITEMS:
        off = item * 2
        # Read uint16 little-endian
        val = buf[off] | (buf[off + 1] << 8)
        if val < 65535:
            val += 1
            buf[off] = val & 0xFF
            buf[off + 1] = (val >> 8) & 0xFF
            
    return bytes(buf)

def query(state, item):
    if time.time() - _START > _TIMEOUT:
        return 0
    
    if 0 <= item < _NUM_ITEMS:
        off = item * 2
        return state[off] | (state[off + 1] << 8)
    else:
        return 0
