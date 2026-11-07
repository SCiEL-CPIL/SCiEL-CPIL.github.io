"""Rewritten program with optimized Count-Min Sketch."""
import array
import time

# Constants for the sketch
ROWS = 4
WIDTH = 128
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647
MAX_UINT16 = 65535

def update(state, item):
    """
    Updates the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int
    returns: bytes (1024 bytes)
    """
    # Convert bytes to array of unsigned short (H)
    # 1024 bytes / 2 bytes per element = 512 elements
    # 4 rows * 128 cols = 512 elements
    counts = array.array('H', state)
    
    for row in range(ROWS):
        p = PRIMES[row]
        # Compute index using the seed's hash structure but with new width
        idx = ((item * p + 31 * row) % MOD_LARGE) % WIDTH
        pos = row * WIDTH + idx
        counts[pos] += 1
        
    return counts.tobytes()

def query(state, item):
    """
    Queries the sketch state for the frequency estimate of an item.
    state: bytes (1024 bytes)
    item: int
    returns: int (non-negative frequency estimate)
    """
    # Convert bytes to array of unsigned short (H)
    counts = array.array('H', state)
    
    min_val = MAX_UINT16
    
    for row in range(ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % WIDTH
        pos = row * WIDTH + idx
        val = counts[pos]
        if val < min_val:
            min_val = val
            
    return min_val
