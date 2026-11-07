"""Dense exact frequency map: 2 bytes per item (uint16) for all 400 items."""
import struct

# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    # Each item gets 2 bytes (uint16 little-endian) at offset item * 2
    # 400 items * 2 bytes = 800 bytes total, fits in 1024
    offset = item * 2
    # Read current count
    current = state[offset] | (state[offset + 1] << 8)
    # Increment
    new_count = current + 1
    # Write back (max 1024, fits easily in uint16)
    state[offset] = new_count & 0xFF
    state[offset + 1] = (new_count >> 8) & 0xFF
    return bytes(state)

def query(state, item):
    # Read the 2-byte counter for this item
    offset = item * 2
    count = state[offset] | (state[offset + 1] << 8)
    return float(count)
# EVOLVE-BLOCK-END
