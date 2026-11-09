import struct

# Constants
W = 256          # width of each row (256 counters)
D = 4            # depth (number of rows)
PRIME = 2147483647

# Pre-compute hash parameters
_PS = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    """
    Update the Count-Min sketch state.
    state: bytes of length 1024 (4 rows * 256 counters * 1 byte per counter)
    item: int in 0..399
    returns: new state as bytes
    """
    # Convert state to a mutable buffer
    buf = bytearray(state)
    
    # For each row, compute the index and increment the counter
    for r in range(D):
        p = _PS[r]
        # Compute index: ((item * p + 31 * r) % PRIME) % W
        idx = ((item * p + 31 * r) % PRIME) % W
        
        # Each counter is 1 byte (uint8)
        offset = r * W + idx
        
        # Read current value
        current = buf[offset]
        
        # Increment (saturating at 255)
        if current < 255:
            buf[offset] = current + 1
        # else: stays 255
        
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    # Use the minimum of the 4 row estimates
    min_est = 256  # Start higher than max uint8
    
    for r in range(D):
        p = _PS[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        
        offset = r * W + idx
        
        # Read the counter value
        val = state[offset]
        
        if val < min_est:
            min_est = val
            
    return int(min_est)
