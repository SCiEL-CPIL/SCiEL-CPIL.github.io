import struct
import time

# Precompute hash parameters for 4 rows, 256 slots each
# State layout: 4 rows × 256 bytes = 1024 bytes
# Each byte is a uint8 counter (0-255)

ROW_SLOTS = 256
NUM_ROWS = 4
STATE_SIZE = NUM_ROWS * ROW_SLOTS  # 1024

# Hash primes for each row - use large odd primes for good distribution
# We need: hash(item, row) = ((item * prime + offset) % large_prime) % 256
# But simpler: use XOR-shift or multiplicative hash directly mod 256

# For 256 slots, we can use: (item * prime + row_salt) % 256
# But this only uses low bits. Better to use a mixing function.

# Simple approach: use different multipliers and take bits
# hash = ((item * mult + add) >> shift) & 255
# Or: use the full value mod a large prime, then mod 256

# Let's use: idx = ((item * P[row] + S[row]) % 2**32) % 256
# This is fast and distributes well

PRIMES = [
    1035493289,  # row 0
    775374203,   # row 1  
    1495862053,  # row 2
    2015145641,  # row 3
]
SALTS = [
    0, 1, 2, 3
]

def _hash(item, row):
    """Compute hash index for item in given row, returns 0..255"""
    return ((item * PRIMES[row] + SALTS[row]) % 4294967296) % 256

def update(state, item):
    """Update state by incrementing counters for item."""
    state = bytearray(state)
    for row in range(NUM_ROWS):
        idx = _hash(item, row)
        pos = row * ROW_SLOTS + idx
        # Saturating increment: if 255, stay at 255
        if state[pos] < 255:
            state[pos] += 1
    return bytes(state)

def query(state, item):
    """Query estimated frequency for item."""
    estimates = []
    for row in range(NUM_ROWS):
        idx = _hash(item, row)
        pos = row * ROW_SLOTS + idx
        estimates.append(state[pos])
    return min(estimates)
