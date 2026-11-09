"""Hybrid exact/sketch frequency estimator using 1024 bytes of state.

Structural design:
  - Items 0..255: exact uint8 counters (256 bytes). This captures the bulk of
    Zipf(1.6) mass, the entire left side of the bimodal stream (0..19), and
    the dead zone (20..149) with zero error.
  - Items 256..399: 4-row x 192-slot Count-Min Sketch with uint8 counters
    (4 * 192 = 768 bytes). Total = 256 + 768 = 1024 bytes.

  For the sketch portion, we use a 4-row CMS with width 192.
  This reduces the expected load per slot compared to the previous design,
  significantly lowering collision-induced overestimation for the tail items.
"""

# Constants
MOD = 2147483647
EXACT_BOUND = 256  # items 0..255 use exact counters
SKETCH_START_ITEM = 256  # items 256..399 use sketch
N_ROWS = 4
W = 192  # width for sketch rows in uint8 counters
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (31, 63, 97, 129)  # offsets for each row to decorrelate indices

# State layout:
#   Bytes 0..255: exact uint8 counters for items 0..255
#   Bytes 256..1023: 4 rows of 192 uint8 counters for items 256..399
#   Total: 1024 bytes


def _sketch_idx(item, row):
    """Compute sketch index for an item in a given row."""
    return (item * PRIMES[row] + OFFSETS[row]) % MOD % W


def update(state, item):
    """Increment the appropriate counter(s) for item in the 1024-byte state."""
    ba = bytearray(state)
    
    if item < EXACT_BOUND:
        # Exact counter: just increment byte at position 'item'
        if ba[item] < 255:
            ba[item] += 1
    else:
        # Use sketch for items >= 256
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = EXACT_BOUND + r * W + idx
            if ba[offset] < 255:
                ba[offset] += 1
    
    return bytes(ba)


def query(state, item):
    """Return nonnegative frequency estimate for item."""
    if item < EXACT_BOUND:
        return float(state[item])
    else:
        best = 255
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = EXACT_BOUND + r * W + idx
            val = state[offset]
            if val < best:
                best = val
        return float(best)
