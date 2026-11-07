import struct
import time

# Constants for the sketch
NUM_ROWS = 4
WIDTH = 128  # 4 * 128 * 2 bytes = 1024 bytes

# Multipliers for hashing to ensure independence between rows
# We need hashes that map item (0..399) to 0..127
# Using a simple linear congruential generator step or just multiplication mod WIDTH
# Since WIDTH is a power of 2, we can use bitwise AND if we generate good bits, 
# but modulo is safer with the given large primes to avoid correlation.

# Let's use the primes from the seed but mod 128
P1 = 1035493289
P2 = 775374203
P3 = 1495862053
P4 = 2015145641

def _update_counters(counters, item):
    # counters is a list of integers
    idx1 = ((item * P1) % 2147483647) % WIDTH
    idx2 = ((item * P2) % 2147483647) % WIDTH
    idx3 = ((item * P3) % 2147483647) % WIDTH
    idx4 = ((item * P4) % 2147483647) % WIDTH
    
    counters[idx1] += 1
    counters[idx2] += 1
    counters[idx3] += 1
    counters[idx4] += 1
    return counters

def update(state, item):
    # Deadline guard
    start_time = time.time()
    
    # Unpack 1024 bytes into 512 uint16s
    # state is bytes, length 1024
    # We expect exactly 1024 bytes.
    if len(state) != 1024:
        # If state is invalid, try to repair or just return zeros? 
        # The contract says initial state is 1024 zero bytes.
        # If it's not, we might be in an error state. 
        # Let's assume it's always 1024.
        pass
        
    # struct.unpack('<512H', state) creates a tuple of 512 unsigned shorts
    counters = list(struct.unpack('<512H', state))
    
    _update_counters(counters, item)
    
    # Pack back to bytes
    new_state = struct.pack('<512H', *counters)
    
    return new_state

def query(state, item):
    # Deadline guard
    start_time = time.time()
    
    counters = struct.unpack('<512H', state)
    
    idx1 = ((item * P1) % 2147483647) % WIDTH
    idx2 = ((item * P2) % 2147483647) % WIDTH
    idx3 = ((item * P3) % 2147483647) % WIDTH
    idx4 = ((item * P4) % 2147483647) % WIDTH
    
    c1 = counters[idx1]
    c2 = counters[idx2]
    c3 = counters[idx3]
    c4 = counters[idx4]
    
    return min(c1, c2, c3, c4)
