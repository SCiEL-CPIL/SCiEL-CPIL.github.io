def update(state, item):
    # state is bytes, item is int
    # Convert to bytearray for mutability
    b = bytearray(state)
    
    # Calculate offset
    offset = item * 2
    
    # Extract current value
    # Little-endian uint16
    current_val = b[offset] | (b[offset+1] << 8)
    
    # Increment
    new_val = current_val + 1
    
    # Write back
    b[offset] = new_val & 0xFF
    b[offset+1] = (new_val >> 8) & 0xFF
    
    return bytes(b)

def query(state, item):
    offset = item * 2
    val = state[offset] | (state[offset+1] << 8)
    return val
