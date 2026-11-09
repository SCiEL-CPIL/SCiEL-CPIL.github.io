"""
Count-Min Sketch with 1 row and 512 width uint16 counters.
Total size: 1 * 512 * 2 bytes = 1024 bytes.
Strategy: Maximize width to minimize bias (N/w), accepting higher variance (no min reduction).
"""

# Configuration
_D = 1
_W = 512

# Hash parameters
# Use a large odd prime-like multiplier for good distribution
_A = [1035493289]
_B = [0]

def update(state, item):
    buf = bytearray(state)
    
    for i in range(_D):
        # Compute index
        idx = (_A[i] * item + _B[i]) % _W
        
        # Calculate byte offset
        byte_offset = i * (_W * 2) + idx * 2
        
        # Read uint16 value
        val = buf[byte_offset] | (buf[byte_offset + 1] << 8)
        
        # Increment
        val += 1
        
        # Write back uint16
        buf[byte_offset] = val & 0xFF
        buf[byte_offset + 1] = (val >> 8) & 0xFF
        
    return bytes(buf)

def query(state, item):
    buf = state
    
    # Since D=1, we just return the single counter
    i = 0
    idx = (_A[i] * item + _B[i]) % _W
    
    byte_offset = i * (_W * 2) + idx * 2
    
    val = buf[byte_offset] | (buf[byte_offset + 1] << 8)
    
    return val
