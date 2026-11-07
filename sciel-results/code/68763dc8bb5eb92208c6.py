"""
Count-Min Sketch with 8 rows x 64 columns of uint16 counters.
State is exactly 1024 bytes.
"""
# EVOLVE-BLOCK-START
import struct

# 8 distinct large odd primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
]
_MOD = 2147483647  # 2^31 - 1
_NUM_ROWS = 8
_NUM_COLS = 64
_TOTAL_COUNTERS = _NUM_ROWS * _NUM_COLS

# Pre-compute indices for all possible items (0..399)
# _INDEXES[item] is a tuple of 8 offsets into the flat 1024-byte buffer
_INDEXES = []
for item in range(400):
    offsets = []
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        # Each counter is 2 bytes (uint16 LE)
        byte_offset = (row * _NUM_COLS + idx) * 2
        offsets.append(byte_offset)
    _INDEXES.append(tuple(offsets))

def update(state, item):
    # Unpack 512 uint16 values
    counters = struct.unpack('<512H', state)
    # Get pre-computed byte offsets for this item
    offsets = _INDEXES[item]
    
    # Convert to list for mutability
    counters = list(counters)
    
    # Increment the 8 counters
    for byte_off in offsets:
        idx_in_counter_array = byte_off // 2
        counters[idx_in_counter_array] += 1
        
    # Pack back
    return struct.pack('<512H', *counters)

def query(state, item):
    # Unpack 512 uint16 values
    counters = struct.unpack('<512H', state)
    offsets = _INDEXES[item]
    
    min_val = 65535
    for byte_off in offsets:
        idx_in_counter_array = byte_off // 2
        val = counters[idx_in_counter_array]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
