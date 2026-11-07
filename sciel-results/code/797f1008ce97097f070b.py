import struct
import sys

# Constants for the sketch
NUM_ROWS = 4
NUM_COLS = 128
STATE_SIZE = NUM_ROWS * NUM_COLS * 2  # 1024 bytes, using 16-bit unsigned integers

# Primes for hashing, same as seed to maintain good distribution properties
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647

def update(state, item):
    """
    Updates the frequency sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes
    """
    # Unpack the state into a list of 16-bit integers
    # struct.unpack is fast enough for 256 items per chunk.
    # Format: 512 unsigned short ('H')
    counters = list(struct.unpack('<512H', state))
    
    # Update the 4 counters corresponding to this item
    for row in range(NUM_ROWS):
        # Calculate index
        # Using the same hash structure as the seed but modulo NUM_COLS
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % NUM_COLS
        
        # Increment the counter
        # Max value is 1024, fits in uint16
        counters[row * NUM_COLS + idx] += 1
        
    # Pack back to bytes
    new_state = struct.pack('<512H', *counters)
    return new_state

def query(state, item):
    """
    Queries the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: non-negative frequency estimate (int)
    """
    counters = struct.unpack('<512H', state)
    
    min_count = 65535 # Max uint16 value
    
    for row in range(NUM_ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % NUM_COLS
        val = counters[row * NUM_COLS + idx]
        if val < min_count:
            min_count = val
            
    return min_count
