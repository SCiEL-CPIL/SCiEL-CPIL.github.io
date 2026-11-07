"""Hybrid direct-count + small CM sketch for streaming frequency estimation."""
import struct

# EVOLVE-BLOCK-START
def update(state, item):
    """Update state with a new item. State is 1024 bytes."""
    # First 400 bytes: uint8 direct counts for items 0..399
    # Next 624 bytes: 4 rows x 39 cols of uint32 CM sketch (4*39*4 = 624)
    
    # Update direct count (saturating at 255)
    direct = state[item]
    if direct < 255:
        new_direct = direct + 1
        state = state[:item] + bytes([new_direct]) + state[item+1:]
    
    # Update CM sketch
    # 4 rows, 39 columns, uint32 little-endian
    # Index: ((item * p + 31 * row) % 2147483647) % 39
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    for row in range(4):
        col = ((item * primes[row] + 31 * row) % 2147483647) % 39
        offset = 400 + row * 39 * 4 + col * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < 0xFFFFFFFF:
            val += 1
        struct.pack_into('<I', state, offset, val)
    
    return state

def query(state, item):
    """Query the estimated frequency of an item."""
    # Direct count
    direct = state[item]
    
    # If not saturated, use direct count (exact)
    if direct < 255:
        return float(direct)
    
    # If saturated, use CM sketch estimate
    primes = (1035493289, 775374203, 1495862053, 2015145641)
    min_val = 0xFFFFFFFF
    for row in range(4):
        col = ((item * primes[row] + 31 * row) % 2147483647) % 39
        offset = 400 + row * 39 * 4 + col * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
    return float(min_val)
# EVOLVE-BLOCK-END
