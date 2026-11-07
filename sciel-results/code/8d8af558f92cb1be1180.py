import struct

# Layout:
# Bytes 0-127: presence bitset (400 bits, 50 bytes used, rest padding)
# Bytes 128-1023: 4 rows x 56 counters, uint32 little-endian (896 bytes)

ROWS = 4
COLS = 56
BITSET_OFFSET = 0
SKETCH_OFFSET = 128
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647

def _bit_index(item):
    """Return byte offset and bit mask for item in the presence bitset."""
    byte_off = item // 8
    bit_mask = 1 << (item % 8)
    return byte_off, bit_mask

def update(state, item):
    state = bytearray(state)
    
    # Set presence bit
    byte_off, bit_mask = _bit_index(item)
    state[BITSET_OFFSET + byte_off] |= bit_mask
    
    # Update 4-row CM sketch
    for r in range(ROWS):
        idx = ((item * PRIMES[r] + 31 * r) % MOD) % COLS
        off = SKETCH_OFFSET + r * COLS * 4 + idx * 4
        val = struct.unpack_from('<I', state, off)[0]
        val += 1
        struct.pack_into('<I', state, off, val)
    
    return bytes(state)

def query(state, item):
    # Check presence bit
    byte_off, bit_mask = _bit_index(item)
    if not (state[BITSET_OFFSET + byte_off] & bit_mask):
        return 0
    
    # Query 4-row CM sketch, return minimum
    min_val = 0xFFFFFFFF
    for r in range(ROWS):
        idx = ((item * PRIMES[r] + 31 * r) % MOD) % COLS
        off = SKETCH_OFFSET + r * COLS * 4 + idx * 4
        val = struct.unpack_from('<I', state, off)[0]
        if val < min_val:
            min_val = val
    
    return min_val
