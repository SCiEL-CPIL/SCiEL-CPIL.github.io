"""Exact frequency table for 400 items using 16-bit counters in 1024-byte state."""
import struct

# 400 items * 2 bytes (uint16) = 800 bytes for exact counts
# Remaining 224 bytes unused (or could hold a small CMS, but exact table suffices)
NUM_ITEMS = 400
COUNTER_SIZE = 2  # uint16 per item
TABLE_SIZE = NUM_ITEMS * COUNTER_SIZE  # 800 bytes

def update(state, item):
    """
    Updates the exact frequency table with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: bytes (1024 bytes)
    """
    b = bytearray(state)
    offset = item * COUNTER_SIZE
    val = struct.unpack_from('<H', b, offset)[0]
    val += 1
    if val > 65535:
        val = 65535
    struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    """
    Queries the exact frequency table for the frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: int (non-negative frequency estimate)
    """
    offset = item * COUNTER_SIZE
    val = struct.unpack_from('<H', state, offset)[0]
    return val
