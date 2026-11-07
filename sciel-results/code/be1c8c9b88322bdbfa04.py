# EVOLVE-BLOCK-START
import struct

_NUM_EXACT = 256          # items 0..255 get exact counters
_NUM_CM_ROWS = 12
_NUM_CM_COLS = 16
_CM_BASE = 512            # byte offset where CM region starts

# Primes for CM rows (12 distinct large odd primes)
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    410332281,
    524287001,
    628318531,
    734003371,
]
_MOD = 2147483647  # 2^31 - 1

# Pre-compute CM byte offsets for items 256..399
_CM_OFFSETS = {}
for item in range(256, 400):
    offsets = []
    for row in range(_NUM_CM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
        byte_off = _CM_BASE + (row * _NUM_CM_COLS + idx) * 2
        offsets.append(byte_off)
    _CM_OFFSETS[item] = tuple(offsets)


def update(state, item):
    if item < _NUM_EXACT:
        # Exact uint16 counter at byte offset item*2
        off = item * 2
        val = struct.unpack_from('<H', state, off)[0]
        val = min(val + 1, 65535)
        buf = bytearray(state)
        struct.pack_into('<H', buf, off, val)
        return bytes(buf)
    else:
        # CM sketch: increment all 12 row counters
        offsets = _CM_OFFSETS[item]
        buf = bytearray(state)
        for off in offsets:
            val = struct.unpack_from('<H', buf, off)[0]
            val = min(val + 1, 65535)
            struct.pack_into('<H', buf, off, val)
        return bytes(buf)


def query(state, item):
    if item < _NUM_EXACT:
        off = item * 2
        return float(struct.unpack_from('<H', state, off)[0])
    else:
        offsets = _CM_OFFSETS[item]
        min_val = 65535
        for off in offsets:
            val = struct.unpack_from('<H', state, off)[0]
            if val < min_val:
                min_val = val
        return float(min_val)
# EVOLVE-BLOCK-END
