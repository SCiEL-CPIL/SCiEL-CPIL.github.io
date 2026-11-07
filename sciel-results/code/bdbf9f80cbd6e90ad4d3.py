def update(state, item):
    """Update the sketch state with a new item."""
    state = bytearray(state)
    
    # Update exact 8-bit counter for this item
    state[item] = (state[item] + 1) % 256
    
    # Update CM sketch: 4 rows x 39 counters of uint32, starting at byte 400
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    for row in range(4):
        idx = ((item * primes[row] + 31 * row) % 2147483647) % 39
        offset = 400 + row * 39 * 4 + idx * 4
        val = int.from_bytes(state[offset:offset+4], 'little')
        val += 1
        state[offset:offset+4] = val.to_bytes(4, 'little')
    
    return bytes(state)

def query(state, item):
    """Query the frequency estimate for an item."""
    # Exact 8-bit count
    exact = state[item]
    
    # CM sketch estimate: min of 4 rows
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    cm_min = 2**32
    for row in range(4):
        idx = ((item * primes[row] + 31 * row) % 2147483647) % 39
        offset = 400 + row * 39 * 4 + idx * 4
        val = int.from_bytes(state[offset:offset+4], 'little')
        if val < cm_min:
            cm_min = val
    
    # If CM estimate is significantly larger, the 8-bit counter likely overflowed
    if cm_min > exact + 8:
        return float(cm_min)
    else:
        return float(exact)
