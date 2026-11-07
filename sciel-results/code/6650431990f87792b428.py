import struct
import sys

# Constants for the sketch
NUM_ROWS = 4
NUM_COLS = 128
STATE_SIZE = NUM_ROWS * NUM_COLS * 2  # 1024 bytes, using 16-bit unsigned integers

# Primes for hashing, same as seed to maintain good distribution properties
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647
MAX_ITEM = 400

# Precompute hash indices for all items and rows to speed up update/query
# indices[row][item] -> column index
_indices = [[0] * MAX_ITEM for _ in range(NUM_ROWS)]
for row in range(NUM_ROWS):
    p = PRIMES[row]
    salt = 31 * row
    for item in range(MAX_ITEM):
        _indices[row][item] = ((item * p + salt) % MOD_LARGE) % NUM_COLS

def update(state, item):
    """
    Updates the frequency sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes
    """
    # Convert to bytearray for mutable updates
    ba = bytearray(state)
    
    # Update the 4 counters corresponding to this item
    for row in range(NUM_ROWS):
        idx = _indices[row][item]
        # Each counter is 2 bytes (little-endian uint16)
        # Offset in bytes: row * (NUM_COLS * 2) + idx * 2
        offset = row * (NUM_COLS * 2) + idx * 2
        
        # Read current value
        val = ba[offset] | (ba[offset + 1] << 8)
        
        # Increment
        val += 1
        
        # Write back
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    """
    Queries the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: non-negative frequency estimate (int)
    """
    min_count = 65535 # Max uint16 value
    
    for row in range(NUM_ROWS):
        idx = _indices[row][item]
        offset = row * (NUM_COLS * 2) + idx * 2
        
        val = state[offset] | (state[offset + 1] << 8)
        if val < min_count:
            min_count = val
            
    return min_count
