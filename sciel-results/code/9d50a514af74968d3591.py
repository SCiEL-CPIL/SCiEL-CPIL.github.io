import struct
import time

# State layout (1024 bytes total):
#   0..1023: 16 rows x 64 columns of uint8 counters (CM sketch)
# All 400 items are handled by the CM sketch.

NUM_ROWS = 16
COUNTERS_PER_ROW = 64

# Hash parameters: use different primes for each row to minimize collisions
# For row r, index = (item * P[r] + S[r]) % 64
# We'll use well-known large primes and salts
_PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    2654435761, 3223224571, 3894226717, 4461868209,
    5031285829, 5601712393, 6173148961, 6745595533,
    7318042109, 7890488681, 8462935253, 9035381827
]
_SALTS = [
    31, 67, 103, 139, 175, 211, 247, 283,
    319, 355, 391, 427, 463, 499, 535, 571
]

def _get_index(item, row):
    return (item * _PRIMES[row] + _SALTS[row]) % COUNTERS_PER_ROW

# Pre-compute positions for all 400 items
# Position in the flat 1024-byte state = row * COUNTERS_PER_ROW + idx
_POSITIONS = {}
for item in range(400):
    positions = []
    for row in range(NUM_ROWS):
        idx = _get_index(item, row)
        pos = row * COUNTERS_PER_ROW + idx
        positions.append(pos)
    _POSITIONS[item] = positions

def update(state, item):
    # Work directly on bytes for speed
    counters = bytearray(state)
    positions = _POSITIONS[item]
    for pos in positions:
        counters[pos] = (counters[pos] + 1) & 0xFF
    return bytes(counters)

def query(state, item):
    counters = state  # bytes is already indexable
    positions = _POSITIONS[item]
    min_val = 255
    for pos in positions:
        val = counters[pos]
        if val < min_val:
            min_val = val
    return min_val
