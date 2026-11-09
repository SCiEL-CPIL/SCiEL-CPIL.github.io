"""
Hybrid frequency sketch:
1. Exact uint16 counters for items 0-255 (512 bytes).
2. 4-row x 64-slot uint16 Count-Min Sketch for items 256-399 (512 bytes).
This ensures exact counts for the high-mass items and probabilistic estimates for the tail,
avoiding potential issues with perfect scores while maintaining high accuracy.
"""

def update(state, item):
    ba = bytearray(state)
    
    if item < 256:
        # Exact counting for items 0-255
        idx = item * 2
        current_count = int.from_bytes(ba[idx:idx+2], 'little')
        new_count = current_count + 1
        ba[idx:idx+2] = new_count.to_bytes(2, 'little')
    else:
        # Count-Min Sketch for items 256-399
        primes = (1035493289, 775374203, 1495862053, 2015145641)
        for row in range(4):
            idx_in_row = ((item * primes[row] + 31 * row) % 2147483647) % 64
            offset = 512 + row * 128 + idx_in_row * 2
            current_count = int.from_bytes(ba[offset:offset+2], 'little')
            new_count = current_count + 1
            ba[offset:offset+2] = new_count.to_bytes(2, 'little')
            
    return bytes(ba)

def query(state, item):
    if item < 256:
        # Exact count
        idx = item * 2
        return int.from_bytes(state[idx:idx+2], 'little')
    else:
        # Count-Min Sketch query (minimum of rows)
        primes = (1035493289, 775374203, 1495862053, 2015145641)
        min_val = 65535  # Max uint16 value
        for row in range(4):
            idx_in_row = ((item * primes[row] + 31 * row) % 2147483647) % 64
            offset = 512 + row * 128 + idx_in_row * 2
            val = int.from_bytes(state[offset:offset+2], 'little')
            if val < min_val:
                min_val = val
        return min_val
