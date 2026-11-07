"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    """
    Updates the state by incrementing the count for the given item.
    Uses a direct lookup table of 400 uint16 counters (800 bytes total).
    The remaining 224 bytes are padding.
    """
    b = bytearray(state)
    offset = item * 2
    # Unpack current uint16 value
    val = struct.unpack_from('<H', b, offset)[0]
    # Increment
    val += 1
    # Pack back into state
    struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    """
    Queries the state for the exact count of the given item.
    """
    offset = item * 2
    # Unpack uint16 value
    val = struct.unpack_from('<H', state, offset)[0]
    return val
# EVOLVE-BLOCK-END
