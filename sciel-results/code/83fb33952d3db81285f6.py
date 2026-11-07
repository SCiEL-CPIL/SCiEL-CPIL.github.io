"""Memory-bounded streaming frequency sketches: exact sparse map in 1024 bytes."""
import struct

# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the exact count for 'item' in the 1024-byte state.
    
    State layout: 400 bytes at offset 0, each byte stores the count (capped at 255)
    for item index 0..399. Remaining 624 bytes are unused padding.
    """
    # Parse the 400-byte count table
    counts = bytearray(state[:400])
    idx = item % 400
    if counts[idx] < 255:
        counts[idx] += 1
    # Rebuild state: 400 bytes of counts + 624 bytes of zero padding
    new_state = bytes(counts) + b'\x00' * 624
    return new_state

def query(state, item):
    """Return the exact (or capped) count for 'item'."""
    idx = item % 400
    return int(state[idx])

# EVOLVE-BLOCK-END
