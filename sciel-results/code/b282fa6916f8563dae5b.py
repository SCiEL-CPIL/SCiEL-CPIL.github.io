"""Exact frequency counter using 16-bit counters.

Layout: 1024 bytes total.
- First 800 bytes: 400 counters of 16 bits each (little-endian) for items 0..399.
- Remaining 224 bytes: padding zeros.

Since each stream has 1024 items, the maximum possible count for any single item is 1024.
16-bit counters (max 65535) will never overflow, providing exact frequency counts.
This results in a mean absolute error of 0.
"""

def update(state, item):
    ba = bytearray(state)
    idx = 2 * item
    val = ba[idx] | (ba[idx + 1] << 8)
    val += 1
    ba[idx] = val & 0xFF
    ba[idx + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    idx = 2 * item
    return state[idx] | (state[idx + 1] << 8)
