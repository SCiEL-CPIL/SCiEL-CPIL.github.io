"""Memory-bounded streaming frequency sketch for items 0..399.

Hybrid approach:
- Bytes 0-127: exact uint8 counters for items 0-127 (captures Zipf head with zero error)
- Bytes 128-1023: 4 rows x 224 uint8 counters (Count-Min sketch) for items 128-399

Total: 128 + 4*224 = 128 + 896 = 1024 bytes
"""
import numpy as np

# Hashing parameters for the CM sketch portion
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
ROWS = 4
SKETCH_WIDTH = 192  # 192 uint8 counters per row
EXACT_COUNT = 256   # Items 0-255 stored exactly
SKETCH_OFFSET = EXACT_COUNT  # Start of sketch in byte array


def update(state, item):
    """Update the sketch with one item. Returns new state as bytes."""
    buf = bytearray(state)
    item = int(item)
    
    if item < EXACT_COUNT:
        # Exact counter for items 0-127
        if buf[item] < 255:
            buf[item] = buf[item] + 1
    else:
        # CM sketch for items 128-399
        # Remap item to 0..171 range for hashing
        remapped = item - EXACT_COUNT
        for row in range(ROWS):
            p = PRIMES[row]
            idx = ((remapped * p + 31 * row) % 2147483647) % SKETCH_WIDTH
            flat_idx = SKETCH_OFFSET + row * SKETCH_WIDTH + idx
            if buf[flat_idx] < 255:
                buf[flat_idx] = buf[flat_idx] + 1
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative integer."""
    item = int(item)
    
    if item < EXACT_COUNT:
        # Exact counter
        return state[item]
    else:
        # CM sketch query: return minimum across rows
        remapped = item - EXACT_COUNT
        min_count = float('inf')
        for row in range(ROWS):
            p = PRIMES[row]
            idx = ((remapped * p + 31 * row) % 2147483647) % SKETCH_WIDTH
            flat_idx = SKETCH_OFFSET + row * SKETCH_WIDTH + idx
            val = state[flat_idx]
            if val < min_count:
                min_count = val
        
        return max(0, min_count)
