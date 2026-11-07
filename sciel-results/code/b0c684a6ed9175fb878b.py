import struct

def update(state, item):
    """
    Update the 4-row Count-Min Sketch.
    State: 1024 bytes (4 rows * 128 counters * 2 bytes/counter).
    """
    b = bytearray(state)
    
    # 4 rows, 128 counters per row (uint16)
    # Row r starts at offset r * 256
    # Counter i in row r is at offset r * 256 + i * 2
    
    # Hash parameters
    # Using strong multipliers to ensure good distribution
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    primes = 2147483647
    width = 128
    
    for r in range(4):
        # Calculate hash index
        # Add a row-specific salt to decorrelate rows
        salt = 31 * (r + 1)
        idx = ((item * multipliers[r] + salt) % primes) % width
        
        offset = r * 256 + idx * 2
        
        # Read current value (little-endian uint16)
        val = int.from_bytes(b[offset:offset+2], 'little')
        
        # Increment
        val += 1
        
        # Write back
        b[offset:offset+2] = val.to_bytes(2, 'little')
        
    return bytes(b)

def query(state, item):
    """
    Query the 4-row Count-Min Sketch.
    Returns the minimum count across all rows.
    """
    # Hash parameters
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    primes = 2147483647
    width = 128
    
    min_val = 1024  # Max possible count
    
    for r in range(4):
        salt = 31 * (r + 1)
        idx = ((item * multipliers[r] + salt) % primes) % width
        
        offset = r * 256 + idx * 2
        
        # Read value
        val = int.from_bytes(state[offset:offset+2], 'little')
        
        if val < min_val:
            min_val = val
            
    return min_val
