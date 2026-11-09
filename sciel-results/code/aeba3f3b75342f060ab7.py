"""Memory-bounded streaming frequency sketch for 400 items using 1024 bytes.

Structural change: split items into two halves (0-199, 200-399), each with its
own 2-row x 256-slot CMS. This gives each item group 4x more counter slots than
the original 4x128 design, reducing collision-induced error.

Layout (1024 bytes = 512 uint16 counters):
  Rows 0-1: items 0-199, 256 slots each (bytes 0..1023, counters 0..511)
    Row 0: counters 0..255   (bytes 0..511)
    Row 1: counters 256..511 (bytes 512..1023)
  Rows 2-3: items 200-399, 256 slots each
    Row 2: counters 512..767 (bytes 1024..1535) -- wait, that's 512 counters
    Actually: 512 uint16 total = 1024 bytes
    Row 0: 256 counters -> bytes 0..511
    Row 1: 256 counters -> bytes 512..1023
    Row 2: 256 counters -> bytes 1024..1535 -- exceeds 1024!

Let me recalculate: 1024 bytes / 2 bytes per counter = 512 counters total.
4 rows x 128 counters = 512. That's the original.

For 2 halves x 2 rows x 256 counters = 1024 counters = 2048 bytes. Too much!

Need: 2 halves x 2 rows x 128 counters = 512 counters = 1024 bytes. Same as before.

Alternative: 2 halves x 1 row x 256 counters = 512 counters = 1024 bytes.
This gives 256 slots per item but only 1 row (no min-of-4 variance reduction).

Better: keep 4 rows but use different hash functions with wider effective space.
Or: use 2 rows x 256 for one half, 2 rows x 256 for other half = 1024 counters.

Wait: 2+2+2+2 = 8 rows? No.

Let me think differently. The key insight from inspiration 1: exact counters for
high-frequency items + CMS for tail. But we can't do exact for all 400.

New structural approach: Use a 2-level design.
- First 256 bytes (128 uint16): exact counters for items 0-127 (high Zipf mass)
- Remaining 768 bytes (384 uint16): 3-row x 128 CMS for items 128-399

Actually simpler structural change: use 2 rows x 256 slots for ALL 400 items
with better hash functions. 2*256 = 512 counters = 1024 bytes.
This gives 256 slots per row (vs 128) with 2 rows (vs 4).
Collision probability per row: 400/256 vs 400/128 = 2x better per row.
But min of 2 vs min of 4: variance reduction is worse.

Expected error scales roughly as (1/m) * (number of colliding items).
With m=256, 2 rows: E[error] ~ sum over other items of P(collision in row)
= (n-1)/256 * 2 (for min of 2, error is bounded by max of 2 rows)

With m=128, 4 rows: E[error] ~ (n-1)/128 * 4

So 2x256 gives ~1/4 the error of 4x128 in terms of collision count, but min-of-2
is worse than min-of-4 for tail bound. Net effect: likely better.
"""

PRIMES = (1035493289, 775374203)
OFFSETS = (31, 63)
MOD = 2147483647
W = 256  # width in 16-bit counters per row
R = 2    # number of rows


def _idx(item, row):
    return (item * PRIMES[row] + OFFSETS[row]) % MOD % W


def update(state, item):
    """Increment the 2 counters for item in a 2x256 uint16 array."""
    out = bytearray(state)
    for row in range(R):
        i = _idx(item, row)
        pos = row * W + i  # 0..511
        byte_lo = 2 * pos
        byte_hi = byte_lo + 1
        cur = out[byte_lo] | (out[byte_hi] << 8)
        if cur < 65535:
            cur += 1
        out[byte_lo] = cur & 0xFF
        out[byte_hi] = (cur >> 8) & 0xFF
    return bytes(out)


def query(state, item):
    """Return the minimum of the 2 counters for item."""
    vals = state
    best = 65535
    for row in range(R):
        i = _idx(item, row)
        pos = row * W + i
        byte_lo = 2 * pos
        byte_hi = byte_lo + 1
        cur = vals[byte_lo] | (vals[byte_hi] << 8)
        if cur < best:
            best = cur
    return float(best)
