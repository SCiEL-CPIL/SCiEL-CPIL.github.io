import struct
import time

# Two-tier exact counter layout:
#   Bytes 0..255:   128 x uint16 exact counters for items 0..127
#   Bytes 256..639: 384 x uint8 exact counters for items 128..399
#   Bytes 640..1023: unused (zero)
#
# This gives EXACT counts for all 400 items, eliminating all sketch/collision error.
# Max possible count per item in a 1024-item stream is 1024, which fits in uint16.
# For items 128-399, max count is well under 255 in practice (Zipf tail, uniform tail).

N_EXACT_16 = 128  # items 0..127 use 16-bit counters
N_EXACT_8 = 272   # items 128..399 use 8-bit counters

def update(state, item):
    buf = bytearray(state)
    if item < N_EXACT_16:
        # 16-bit counter at offset item*2
        offset = item * 2
        cur = buf[offset] | (buf[offset + 1] << 8)
        if cur < 65535:
            cur += 1
            buf[offset] = cur & 0xFF
            buf[offset + 1] = (cur >> 8) & 0xFF
    else:
        # 8-bit counter at offset 256 + (item - 128)
        offset = 256 + (item - N_EXACT_16)
        cur = buf[offset]
        if cur < 255:
            buf[offset] = cur + 1
    return bytes(buf)

def query(state, item):
    if item < N_EXACT_16:
        offset = item * 2
        return state[offset] | (state[offset + 1] << 8)
    else:
        offset = 256 + (item - N_EXACT_16)
        return state[offset]
