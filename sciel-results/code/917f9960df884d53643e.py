"""
Improved Count-Min Sketch using uint16 counters.
Configuration: Depth 2, Width 256.
Total size: 2 * 256 * 2 bytes = 1024 bytes.
"""

# Configuration
_D = 2
_W = 256

# Hash parameters
# Reusing the prime multipliers from the seed for good distribution
_A = [1035493289, 775374203]
# Offsets to decorrelate rows
_B = [0, 129]

def update(state, item):
    # Convert bytes to mutable bytearray
    buf = bytearray(state)
    
    for i in range(_D):
        # Compute index
        idx = (_A[i] * item + _B[i]) % _W
        
        # Calculate byte offset
        # Row i starts at i * _W * 2
        # Counter idx is at offset idx * 2 within the row
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
    # state is bytes
    buf = state
    
    min_val = 65535  # Max value for uint16
    
    for i in range(_D):
        idx = (_A[i] * item + _B[i]) % _W
        
        byte_offset = i * (_W * 2) + idx * 2
        
        val = buf[byte_offset] | (buf[byte_offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
