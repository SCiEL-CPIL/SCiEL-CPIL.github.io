"""Memory-bounded streaming frequency sketch: direct 400×uint16 counter table."""
import struct

def update(state, item):
    """Increment the counter for the given item. State is 1024 bytes:
    first 800 bytes = 400 little-endian uint16 counters (items 0..399),
    remaining 224 bytes unused/padding."""
    # Read current count
    offset = item * 2
    current = struct.unpack_from('<H', state, offset)[0]
    new_count = current + 1
    # Write back (cap at 65535 to avoid overflow, though max is 1024)
    new_state = bytearray(state)
    struct.pack_into('<H', new_state, offset, min(new_count, 65535))
    return bytes(new_state)

def query(state, item):
    """Return the exact frequency estimate for the item."""
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    return float(count)
