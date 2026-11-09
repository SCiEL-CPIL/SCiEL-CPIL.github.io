"""Memory-bounded streaming frequency sketches: Fixed-Window Exact + CM tail."""
# EVOLVE-BLOCK-START
import struct
import array

# Layout:
# Bytes 0..255: exact counters for items 0..255 (1 byte each, capped at 255)
# Bytes 256..1023: 4 rows x 192 slots x uint16 = 4*192*2 = 1536? No, 4*192*2=1536 > 768.
# Let's recalculate: 1024 - 256 = 768 bytes for sketch.
# 4 rows. 768 / 4 = 192 bytes per row.
# If uint16, 192 bytes = 96 slots per row.
# So: 4 rows x 96 slots x uint16.
# Total sketch bytes: 4 * 96 * 2 = 768. Correct.

SKETCH_ROWS = 2
SKETCH_COLS = 384
SKETCH_OFFSET = 256
SKETCH_STRIDE = SKETCH_COLS  # 384 bytes per row

# Hash constants for the tail sketch (items 256..399)
# We need good mixing for the 144 tail items.
# Using large primes to avoid correlation.
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    """
    Update the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes (1024 bytes)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        # Fallback: return zero state if invalid
        return b'\x00' * 1024
    
    # Convert to bytearray for mutation
    buf = bytearray(state)
    
    if item < 256:
        # Exact counter
        idx = item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
        # else: cap at 255
    else:
        # Tail item: use CM sketch
        # Item range 256..399.
        # The hash function should map item to 0..191 for each row.
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            # Hash: ((item * p + 31 * row) % 2147483647) % SKETCH_COLS
            h = ((item * p + 31 * row) % 2147483647) % SKETCH_COLS
            
            # Calculate offset in buffer
            # Row r starts at SKETCH_OFFSET + r * SKETCH_STRIDE
            # Slot h is at offset + h (uint8)
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + h
            
            # Read uint8
            val = buf[offset]
            
            # Increment
            if val < 255:
                buf[offset] = val + 1
            # else: cap at 255
    
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative frequency estimate (int or float)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0
    
    if item < 256:
        # Exact count
        return state[item]
    else:
        # CM sketch query: min over rows
        min_val = 255
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            h = ((item * p + 31 * row) % 2147483647) % SKETCH_COLS
            
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + h
            
            val = state[offset]
            
            if val < min_val:
                min_val = val
        
        return min_val
# EVOLVE-BLOCK-END
