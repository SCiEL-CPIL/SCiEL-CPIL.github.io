import struct

# Constants
W_EXACT = 200     # Number of exact counters (items 0-199)
W_SKETCH = 206    # Width of each sketch row (824 bytes / 4 rows)
D_SKETCH = 4      # Depth of sketch
PRIME = 2147483647

# Pre-compute hash parameters
_PS = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    """
    Update the state.
    Items 0-199 use exact 1-byte counters.
    Items 200-399 use a 4x206 Count-Min Sketch.
    """
    buf = bytearray(state)
    
    if item < W_EXACT:
        # Exact counter
        current = buf[item]
        if current < 255:
            buf[item] = current + 1
    else:
        # Sketch counter
        for r in range(D_SKETCH):
            p = _PS[r]
            # Compute index in 0..205
            idx = ((item * p + 31 * r) % PRIME) % W_SKETCH
            offset = W_EXACT + r * W_SKETCH + idx
            
            current = buf[offset]
            if current < 255:
                buf[offset] = current + 1

    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency.
    """
    if item < W_EXACT:
        return int(state[item])
    else:
        min_est = 256
        for r in range(D_SKETCH):
            p = _PS[r]
            idx = ((item * p + 31 * r) % PRIME) % W_SKETCH
            offset = W_EXACT + r * W_SKETCH + idx
            val = state[offset]
            if val < min_est:
                min_est = val
        return int(min_est)
