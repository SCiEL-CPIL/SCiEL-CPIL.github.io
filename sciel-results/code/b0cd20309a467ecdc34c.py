import struct

NUM_ROWS = 8
NUM_COLS = 64

A_PARAMS = [
    1035493289, 775374203, 1495862053, 2015145641,
    3037000493, 2292478453, 180811543, 15485863
]
B_PARAMS = list(range(8))

def update(state, item):
    ba = bytearray(state)
    for r in range(NUM_ROWS):
        idx = ((A_PARAMS[r] * item + B_PARAMS[r]) & 0xFFFFFFFF) % NUM_COLS
        pos = r * (NUM_COLS * 2) + idx * 2
        val = ba[pos] | (ba[pos + 1] << 8)
        val += 1
        ba[pos] = val & 0xFF
        ba[pos + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    min_val = 65535
    for r in range(NUM_ROWS):
        idx = ((A_PARAMS[r] * item + B_PARAMS[r]) & 0xFFFFFFFF) % NUM_COLS
        pos = r * (NUM_COLS * 2) + idx * 2
        val = state[pos] | (state[pos + 1] << 8)
        if val < min_val:
            min_val = val
    return min_val
