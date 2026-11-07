import struct
import time

_START = time.monotonic()
_DEADLINE = _START + 145.0  # Safety margin before 150s kill

# Constants
NUM_ROWS = 8
NUM_COLS = 64
STATE_SIZE = 1024
M = 2147483647  # 2^31 - 1

# 8 distinct large primes for hashing
# These are chosen to be large and distinct to minimize correlation
PRIMES = (
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    2654435761,
    3221225473,
    3837387529,
    4450568629
)

def update(state, item):
    """
    Update the Count-Min Sketch state with a new item.
    State is 1024 bytes: 8 rows of 64 uint16 counters.
    """
    if len(state) != STATE_SIZE:
        return b'\x00' * STATE_SIZE
    
    # Check deadline
    if time.monotonic() > _DEADLINE:
        return state

    # Unpack into a list of lists for easy manipulation
    # Each row is 64 uint16s
    cm = []
    for row in range(NUM_ROWS):
        row_data = []
        for col in range(NUM_COLS):
            offset = row * NUM_COLS * 2 + col * 2
            val = struct.unpack_from('<H', state, offset)[0]
            row_data.append(val)
        cm.append(row_data)
    
    # Update the sketch
    for row in range(NUM_ROWS):
        # Hash function: ((item * prime + 31 * row) % M) % NUM_COLS
        col = ((item * PRIMES[row] + 31 * row) % M) % NUM_COLS
        current_val = cm[row][col]
        # Increment, capping at 65535 to prevent overflow (though 1024 items won't reach this)
        if current_val < 65535:
            cm[row][col] = current_val + 1
    
    # Pack back to bytes
    new_state = bytearray(STATE_SIZE)
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            offset = row * NUM_COLS * 2 + col * 2
            struct.pack_into('<H', new_state, offset, cm[row][col])
    
    return bytes(new_state)

def query(state, item):
    """
    Query the estimated frequency of an item.
    Returns the minimum count across all rows.
    """
    if len(state) != STATE_SIZE:
        return 0.0
    
    # Check deadline
    if time.monotonic() > _DEADLINE:
        return 0.0

    min_est = 65535
    
    for row in range(NUM_ROWS):
        col = ((item * PRIMES[row] + 31 * row) % M) % NUM_COLS
        offset = row * NUM_COLS * 2 + col * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_est:
            min_est = val
            
    return float(min_est)
