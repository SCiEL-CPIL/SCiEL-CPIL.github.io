"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    # 4 rows, 256 columns, uint8 counters
    # Total size: 4 * 256 * 1 = 1024 bytes
    b = bytearray(state)
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    
    for row in range(4):
        p = p_values[row]
        # Hash to 0..255
        idx = ((item * p + 31 * row) % 2147483647) % 256
        # Offset in bytes: row * 256 + idx
        offset = row * 256 + idx
        b[offset] = (b[offset] + 1) & 0xFF
        
    return bytes(b)

def query(state, item):
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    min_val = 255
    
    for row in range(4):
        p = p_values[row]
        idx = ((item * p + 31 * row) % 2147483647) % 256
        offset = row * 256 + idx
        val = state[offset]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
