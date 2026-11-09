"""
Hybrid sketch: DAT for hot items 0..63 (uint16), 4-row x 224-col uint8 CM sketch for items 64..399.
Total: 64*2 + 4*224*1 = 128 + 896 = 1024 bytes.
"""

_DAT_SIZE = 64
_DAT_BYTES = _DAT_SIZE * 2  # 128 bytes
_CM_DEPTH = 4
_CM_WIDTH = 224
_CM_BYTES = _CM_DEPTH * _CM_WIDTH  # 896 bytes (uint8)
_TOTAL = _DAT_BYTES + _CM_BYTES  # 1024

# Hash parameters for 4 rows
_A = (1035493289, 775374203, 1495862053, 2015145641)
_B = (31, 97, 163, 229)  # distinct offsets per row


def update(state, item):
    buf = bytearray(state)
    if item < _DAT_SIZE:
        off = item * 2
        v = buf[off] | (buf[off + 1] << 8)
        v += 1
        buf[off] = v & 0xFF
        buf[off + 1] = (v >> 8) & 0xFF
    else:
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            off = _DAT_BYTES + i * _CM_WIDTH + idx
            buf[off] = (buf[off] + 1) & 0xFF
    return bytes(buf)


def query(state, item):
    if item < _DAT_SIZE:
        off = item * 2
        return state[off] | (state[off + 1] << 8)
    else:
        mn = 255
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            off = _DAT_BYTES + i * _CM_WIDTH + idx
            v = state[off]
            if v < mn:
                mn = v
        return mn
