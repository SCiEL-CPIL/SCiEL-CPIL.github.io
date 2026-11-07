import struct

# 8 rows of 128 uint8 counters
# Total bytes: 8 * 128 * 1 = 1024

# Hash parameters
# We use (item * A_i + B_i) % 128
# A_i must be odd for bijectivity mod 128
A = [1, 3, 5, 9, 11, 13, 17, 19]
B = [0, 1, 2, 3, 4, 5, 6, 7]

def update(state, item):
    b = bytearray(state)
    
    for i in range(8):
        bucket = (item * A[i] + B[i]) % 128
        offset = i * 128 + bucket
        val = b[offset] + 1
        b[offset] = val  # uint8 wraps around, but we expect no overflow
        
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    
    min_val = 255 # Max uint8
    
    for i in range(8):
        bucket = (item * A[i] + B[i]) % 128
        offset = i * 128 + bucket
        val = b[offset]
        
        if val < min_val:
            min_val = val
            
    return min_val
