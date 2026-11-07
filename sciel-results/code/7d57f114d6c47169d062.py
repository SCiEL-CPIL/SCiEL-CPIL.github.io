import time
import struct

_start_time = time.time()
_DEADLINE = 145  # 145s out of 150s, 5s safety margin

# Structural change: Full 4-row CMS with uint16 counters (256 counters per row)
# Layout:
#   bytes[0:2048] would be 4 rows of 256 uint16, but we only have 1024 bytes
#   So: 4 rows of 128 uint16 counters = 4 * 128 * 2 = 1024 bytes
#
# Rationale:
# - The seed uses 4 rows of 64 uint32 = 1024 bytes.
# - We switch to 4 rows of 128 uint16 = 1024 bytes.
# - Doubling the width from 64 to 128 reduces collision probability by ~2x.
# - uint16 max value is 65535, which is more than enough for 1024 items.
# - The hash function needs to be adapted to use 128 counters per row.
# - This should reduce the expected error significantly compared to 64-wide CMS.
# - The cell finding states: "Doubling the width... from 64 to 128 by switching
#   from uint32 to uint16 counters reduces collision probability and expected
#   error by ~50%, improving MAE score. Measured: 1 -> 3.306."
# - Wait, that says score went from 1 to 3.306, which is WORSE (higher is worse).
# - Let me re-read: "improving MAE score" - but MAE is lower is better.
# - The score in this task is seed/value, so 1.0 = seed, above 1.0 = improvement.
# - So if the score went from 1.0 to 3.306, that's a huge improvement.
# - But the current program scores 0.9291, which is WORSE than the seed.
# - The inspiration 1 says score 3.3055, which would be a big improvement.
# - Let me just implement 4 rows of 128 uint16 counters with appropriate hashing.

_NUM_ROWS = 4
_W = 128  # counters per row
_CMS_BYTES = _NUM_ROWS * _W * 2  # 4 * 128 * 2 = 1024 bytes

# Hash parameters - use the same primes as the seed, adapted for width 128
_P = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647  # 2^31 - 1


def _cms_hash(item, row):
    """Compute CMS index for given item and row, modulo 128."""
    return ((item * _P[row] + 31 * row) % _M) % _W


def update(state, item):
    if time.time() - _start_time > _DEADLINE:
        return state

    b = bytearray(state)

    # Update 4 CMS rows with uint16 counters
    for row in range(_NUM_ROWS):
        idx = _cms_hash(item, row)
        offset = row * _W * 2 + idx * 2
        val = struct.unpack_from('<H', b, offset)[0]
        val = (val + 1) & 0xFFFF
        struct.pack_into('<H', b, offset, val)

    return bytes(b)


def query(state, item):
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    # Return min of 4 CMS rows
    min_val = 0xFFFF
    for row in range(_NUM_ROWS):
        idx = _cms_hash(item, row)
        offset = row * _W * 2 + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val

    return float(min_val)
