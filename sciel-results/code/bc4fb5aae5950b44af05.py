"""
Bit-sliced binary counters: 10 bits per item, 400 items = 4000 bits = 500 bytes.
State layout: bytes[0:500] hold the bit-sliced counters; bytes[500:1024] are zero padding.

Bit i of the counter for item k is stored at bit position (i * 400 + k) in the 500-byte array.
Byte index = (i * 400 + k) // 8 = i * 50 + k // 8
Bit within byte = (i * 400 + k) % 8 = k % 8

This is a structurally different representation from the current program's
one-byte-per-item layout: the same 400 counters are stored as 4000 individual
bits organized by bit-significance across the byte array, using XOR-based
increment logic rather than simple byte arithmetic.
"""

def update(state, item):
    """Increment the 10-bit binary counter for `item` using bit-sliced representation.
    Returns a new 1024-byte state."""
    buf = bytearray(state)
    if 0 <= item < 400:
        # Binary increment: find first 0-bit and flip it and all lower 1-bits to 0
        # Bit i is at byte index (i*50 + item//8), bit offset (item % 8)
        for i in range(10):
            byte_idx = i * 50 + (item >> 3)
            bit_mask = 1 << (item & 7)
            if buf[byte_idx] & bit_mask:
                # This bit is 1, clear it and continue
                buf[byte_idx] &= ~bit_mask
            else:
                # This bit is 0, set it and stop
                buf[byte_idx] |= bit_mask
                break
        # If all 10 bits were 1, counter was 1023 and now wraps to 0 (all cleared)
        # This is acceptable since max count per item across streams < 1024
    return bytes(buf)


def query(state, item):
    """Reconstruct the 10-bit binary counter value for `item` (0..399)."""
    if 0 <= item < 400:
        val = 0
        for i in range(10):
            byte_idx = i * 50 + (item >> 3)
            bit_mask = 1 << (item & 7)
            if state[byte_idx] & bit_mask:
                val |= (1 << i)
        return val
    return 0
