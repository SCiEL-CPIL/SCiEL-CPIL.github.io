import struct
import time

# State layout (1024 bytes total):
#   0..127:    64 x uint16 exact counters for items 0..63
#   128..1023: 448 x uint16 CM sketch counters (7 rows x 64 counters) for items 64..399

NUM_EXACT = 64
NUM_ROWS = 7
COUNTERS_PER_ROW = 64
CM_BASE = NUM_EXACT  # offset in the flat state array (counter index, not byte)
TOTAL_COUNTERS = NUM_EXACT + NUM_ROWS * COUNTERS_PER_ROW  # 64 + 448 = 512

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z

def _get_index(item, row):
    seed = (item * 1035493289 + row * 775374203) & 0xFFFFFFFFFFFFFFFF
    return _splitmix64(seed) % COUNTERS_PER_ROW

# Pre-compute positions for items 64..399 in the CM portion
_CM_POSITIONS = {}
for item in range(NUM_EXACT, 400):
    positions = []
    for row in range(NUM_ROWS):
        idx = _get_index(item, row)
        pos = CM_BASE + row * COUNTERS_PER_ROW + idx
        positions.append(pos)
    _CM_POSITIONS[item] = positions

def update(state, item):
    # Unpack all 512 uint16 counters
    counters = list(struct.unpack('512H', state))
    
    if item < NUM_EXACT:
        # Exact count for items 0..63
        counters[item] += 1
    else:
        # CM sketch for items 64..399
        positions = _CM_POSITIONS[item]
        for pos in positions:
            counters[pos] += 1
    
    return struct.pack('512H', *counters)

def query(state, item):
    # Unpack all 512 uint16 counters
    counters = struct.unpack('512H', state)
    
    if item < NUM_EXACT:
        # Exact count
        return counters[item]
    else:
        # CM sketch: return minimum
        positions = _CM_POSITIONS[item]
        min_val = 65535
        for pos in positions:
            val = counters[pos]
            if val < min_val:
                min_val = val
        return min_val
