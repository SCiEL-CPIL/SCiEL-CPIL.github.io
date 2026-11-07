"""
Full 16x64 Count-Min Sketch with uint16 counters.
State layout (1024 bytes):
- 16 rows x 64 columns of little-endian uint16 counters = 16*64*2 = 2048... wait, that's 2048 bytes.
  Actually 16*64 = 1024 counters. If uint16, that's 2048 bytes. Too many.
  
Let me reconsider: 1024 bytes total.
- Option A: 8 rows x 64 cols of uint16 = 8*64*2 = 1024 bytes. 8 rows, 16-bit.
- Option B: 16 rows x 32 cols of uint16 = 16*32*2 = 1024 bytes. 16 rows, 32 cols.
- Option C: 4 rows x 64 cols of uint16 + 4 rows x 64 cols of uint8 = 4*64*2 + 4*64*1 = 512+256=768... no.
- Option D: 16 rows x 64 cols of uint8 = 16*64*1 = 1024 bytes. 16 rows, 8-bit. Max count 255.

For 1024 items, max count for any item in uniform stream is ~1024/400 ≈ 2.56, so max is probably ~7-8.
For Zipf(1.6), the top item could have count ~1024 * (1/1)^(1/1.6) / sum... let me estimate.
Zipf(1.6) on 400 items: P(i) = 1/(i^1.6 * H(400,1.6)) where H is the generalized harmonic number.
H(400,1.6) ≈ 400^(1-1.6)/(1-1.6) = 400^(-0.5)/(-0.5) = -1/(0.5*sqrt(400)) = -1/10 = ... 
Actually H(n,s) for s>1 converges. For s=1.6, H(400,1.6) ≈ 1 + 2^-1.6 + 3^-1.6 + ... 
The first few terms: 1, 0.33, 0.19, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03...
Sum ≈ 1 + 0.33 + 0.19 + 0.13 + 0.10 + 0.08 + 0.06 + 0.05 + 0.04 + 0.03 + ... ≈ 2.1 (rough)
So P(1) ≈ 1/2.1 ≈ 0.48, P(2) ≈ 0.33/2.1 ≈ 0.16, P(3) ≈ 0.19/2.1 ≈ 0.09...
Expected count for item 0 (rank 1): 1024 * 0.48 ≈ 491. That exceeds 255!

So uint8 will saturate for the Zipf heavy hitter. We need uint16 for at least some counters.

Let's go with 8 rows x 64 cols of uint16 = 1024 bytes. 8 rows gives decent min-based variance reduction.
"""
import struct
import time

# State: 8 rows x 64 columns of little-endian uint16
_ROWS = 8
_COLS = 64
_SIZE = _ROWS * _COLS  # 512 uint16 counters = 1024 bytes

# Hash: ((item * p + salt) % MOD) % 64
# We need 8 primes and 8 salts optimized for items 0..399
# Using large odd primes and distinct salts
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
_SALTS = [
    31, 57, 113, 199, 271, 389, 467, 541,
]
_MOD = 2147483647  # 2^31 - 1
_MAX_VAL = 65535

def update(state, item):
    buf = bytearray(state)
    for row in range(_ROWS):
        p = _PRIMES[row]
        s = _SALTS[row]
        idx = ((item * p + s) % _MOD) % _COLS
        offset = row * _COLS * 2 + idx * 2
        # Read uint16
        val = struct.unpack_from('<H', buf, offset)[0]
        if val < _MAX_VAL:
            val += 1
        struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    best = _MAX_VAL
    for row in range(_ROWS):
        p = _PRIMES[row]
        s = _SALTS[row]
        idx = ((item * p + s) % _MOD) % _COLS
        offset = row * _COLS * 2 + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < best:
            best = val
    return best
