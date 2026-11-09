"""Memory-bounded streaming frequency sketch for items 0..399.

Uses a Count-Min Sketch with 4 rows x 128 columns of uint16 counters (1024 bytes).
Each row uses a different prime for hashing. Query returns the minimum across rows.
"""
import numpy as np

# Hashing parameters
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
ROWS = 4
WIDTH = 128
TOTAL_SIZE = ROWS * WIDTH * 2  # 1024 bytes (uint16)


def update(state, item):
    """Update the sketch with one item. Returns new state as bytes."""
    # Load state into numpy array (view, not copy for performance)
    buf = bytearray(state)
    # Interpret as array of uint16
    arr = np.frombuffer(buf, dtype=np.uint16).copy()
    
    item = int(item)
    for row in range(ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % 2147483647) % WIDTH
        flat_idx = row * WIDTH + idx
        # Increment with saturation at uint16 max
        if arr[flat_idx] < 65535:
            arr[flat_idx] = arr[flat_idx] + 1
    
    return arr.tobytes()


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative integer."""
    buf = bytearray(state)
    arr = np.frombuffer(buf, dtype=np.uint16)
    
    item = int(item)
    min_count = float('inf')
    for row in range(ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % 2147483647) % WIDTH
        flat_idx = row * WIDTH + idx
        val = int(arr[flat_idx])
        if val < min_count:
            min_count = val
    
    return max(0, min_count)
