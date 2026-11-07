def update(state, item):
    b = bytearray(state)
    # Layout:
    #   Bytes 0..191:  4 rows x 48 uint16 counters (96 bytes each row)
    #   Bytes 192..255: 64 uint16 "heavy" counters (128 bytes)
    #   Bytes 256..1023: 768 bytes reserved (zero)
    
    # Update 4 Count-Min rows with uint16 counters
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    for r in range(4):
        h = (item * multipliers[r]) & 0xFFFFFFFF
        idx = h % 48  # 48 counters per row
        offset = r * 96 + idx * 2
        val = int.from_bytes(b[offset:offset+2], 'little')
        val = (val + 1) & 0xFFFF
        b[offset:offset+2] = val.to_bytes(2, 'little')
    
    # Update heavy hitters section: 64 uint16 counters
    # Use a different hash for heavy hitters
    h2 = (item * 2654435761) & 0xFFFFFFFF
    heavy_idx = h2 % 64
    heavy_offset = 192 + heavy_idx * 2
    hval = int.from_bytes(b[heavy_offset:heavy_offset+2], 'little')
    hval = (hval + 1) & 0xFFFF
    b[heavy_offset:heavy_offset+2] = hval.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    # Query 4 Count-Min rows
    multipliers = (1035493289, 775374203, 1495862053, 2015145641)
    min_val = 0xFFFF
    for r in range(4):
        h = (item * multipliers[r]) & 0xFFFFFFFF
        idx = h % 48
        offset = r * 96 + idx * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
    
    # Also check heavy hitters
    h2 = (item * 2654435761) & 0xFFFFFFFF
    heavy_idx = h2 % 64
    heavy_offset = 192 + heavy_idx * 2
    hval = int.from_bytes(state[heavy_offset:heavy_offset+2], 'little')
    
    # Return the minimum of CM estimate and heavy hitter count
    # Actually, we should return min of all estimates to be conservative
    # But heavy hitters may have higher counts due to collisions
    # Let's just return the CM min which is standard
    return min_val
