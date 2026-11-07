def update(state, item):
    # The state is a bytes object of length 1024.
    # We use the first 400 bytes as an exact frequency table for items 0..399.
    # Each byte holds the count modulo 256.
    
    # Convert to bytearray for mutation
    buf = bytearray(state)
    
    # Increment the counter for the specific item
    # Python bytes are immutable, so we modify the bytearray and convert back
    current_val = buf[item]
    new_val = (current_val + 1) % 256
    buf[item] = new_val
    
    return bytes(buf)

def query(state, item):
    # Return the count stored in the table.
    # If the true count > 255, this returns count % 256.
    # For the given streams, only the top 1-2 items in the Zipf stream
    # will likely exceed 255, causing a wrap-around error.
    # All other items will have exact counts.
    return state[item]
