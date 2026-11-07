"""
Bit-sliced exact counters: 10 bits per item, 400 items = 4000 bits = 500 bytes.
Each item's count is stored in 10 bits (bits 0-9), supporting counts 0..1023.
Update: read the 10-bit counter, increment, write back.
Query: read the 10-bit counter, return the value.
This gives exact counts with zero estimation error.
"""

_NUM_ITEMS = 400
_BITS_PER_ITEM = 10
_BYTES_PER_ITEM = 2  # 10 bits needs 2 bytes (we use 16 bits, only lower 10 matter)
_TOTAL_BYTES = _NUM_ITEMS * _BYTES_PER_ITEM  # 800 bytes, well within 1024


def update(state, item):
    buf = bytearray(state)
    offset = item * _BYTES_PER_ITEM
    # Read current 16-bit value (little-endian)
    val = buf[offset] | (buf[offset + 1] << 8)
    # Increment, mask to 10 bits
    val = (val + 1) & 0x3FF
    # Write back
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)


def query(state, item):
    offset = item * _BYTES_PER_ITEM
    val = state[offset] | (state[offset + 1] << 8)
    return val & 0x3FF
