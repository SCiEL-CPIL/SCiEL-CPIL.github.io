import hashlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Serialize the vector to bytes
    # vector is expected to be a tuple of integers 0, 1, 2
    b = bytes(vector)
    
    # Compute SHA256 hash
    h = hashlib.sha256(b).digest()
    
    # Take the first 8 bytes (64 bits) to create a large integer
    # This provides a high-quality random ordering with negligible collision risk
    val = int.from_bytes(h[:8], 'big')
    
    # Return as float as per the interface requirement
    return float(val)
# EVOLVE-BLOCK-END
