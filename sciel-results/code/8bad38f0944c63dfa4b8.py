"""Exact counting sketch: 400 uint16 counters in 1024-byte state."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    # state is 1024 bytes; first 800 bytes = 400 uint16 LE counters
    # Read current count for this item
    offset = item * 2
    current = struct.unpack_from('<H', state, offset)[0]
    new_count = current + 1
    # Write back
    new_state = bytearray(state)
    struct.pack_into('<H', new_state, offset, new_count)
    return bytes(new_state)

def query(state, item):
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]
# EVOLVE-BLOCK-END
