import struct

# Strategy: Fixed-Window Exact Counter for items 0..255 + Large CMS for 256..399
#
# Layout:
# Bytes 0..255:   uint8 exact counters for items 0..255 (256 items)
# Bytes 256..1023: 4 rows x 192 columns of uint16 counters for items 256..399
#
# Rationale:
# 1. Items 0..255 capture the high-frequency Zipf head (items 0..15 have highest prob)
#    and the entire left mode of the bimodal stream (0..19).
#    Max count for any item in 1024 inserts:
#    - Uniform: ~2.5
#    - Zipf(1.6): Item 0 has ~1/1024 * sum(1/i^1.6)^-1. H(1.6, 400) ~ 1.35. P(0) ~ 1/1.35 ~ 0.74? No, Zipf is normalized.
#      Actually, for Zipf(1.6) on 0..399, the top item gets a significant chunk.
#      Let's estimate max count. If P(0) is high, say 5%, count is ~50. uint8 (255) is safe.
#    - Bimodal: 512 items on 0..19 (20 items). Each gets ~25 on average. Max likely < 50.
#    So uint8 is sufficient for items 0..255.
#
# 2. Items 256..399 (144 items) are handled by the CMS.
#    Remaining bytes: 1024 - 256 = 768 bytes.
#    768 / 2 = 384 uint16 counters.
#    Configuration: 4 rows x 96 columns? Or 4 rows x 96?
#    Wait, 4 * 96 * 2 = 768. Yes.
#    Let's re-calculate.
#    If I use 4 rows x 96 columns, total counters = 384.
#    Load factor for 144 items into 96 bins per row: 144/96 = 1.5.
#    This is better than the previous 144/90 = 1.6.
#    
#    Can we do better?
#    If I reduce the exact window to 240 items (0..239), I get 240 bytes.
#    Remaining: 784 bytes -> 392 counters -> 4 rows x 98 cols.
#    But items 240..255 would go to sketch.
#    The bimodal stream has items 0..19 and 150..399.
#    So items 240..255 are in the right mode of the bimodal stream.
#    If I move 240..255 to the sketch, they become part of the 144+16=160 items in the sketch.
#    Load factor: 160/98 = 1.63. Worse.
#    
#    If I increase exact window to 256 items, I have 144 items in sketch.
#    4 rows x 96 cols. Load factor 144/96 = 1.5.
#    
#    What if I use 3 rows?
#    3 rows x 128 cols = 384 counters.
#    Load factor 144/128 = 1.125. Better collision resistance?
#    But fewer rows means higher variance in CMS estimate (min of 3 vs min of 4).
#    Usually, more rows is better for bias, but wider width is better for variance.
#    Standard CMS theory suggests optimal width W = e * d (items) / rows? No.
#    For fixed space, d=4, W=96 is a reasonable balance.
#    
#    Let's stick with 4 rows x 96 columns.
#    Wait, 4 * 96 * 2 = 768. Correct.
#
# Hashing:
# Map item 256..399 to local index 0..143.
# Hash to 0..95.
# Use simple linear congruential hashes with different primes/constants.

_EXACT_SIZE = 256
_SKETCH_START = 256
_SKETCH_END = 400
_SKETCH_DEPTH = 4
_SKETCH_WIDTH = 96

# Byte offsets
_EXACT_OFFSET = 0
_SKETCH_OFFSET = _EXACT_SIZE  # 256

# Total check: 256 + 4*96*2 = 256 + 768 = 1024. Correct.

# Hash parameters for the sketch
# We need to map local_item (0..143) to 0..95.
# h(x, r) = (a_r * x + b_r) % 96
# Using large primes to decorrelate rows.
_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [31, 97, 193, 311]

def _sketch_hash(item, row):
    local_item = item - _SKETCH_START
    return (_A[row] * local_item + _B[row]) % _SKETCH_WIDTH

def _read_u16(buf, off):
    return buf[off] | (buf[off+1] << 8)

def _write_u16(buf, off, val):
    buf[off] = val & 0xFF
    buf[off+1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    if item < _EXACT_SIZE:
        # Exact uint8 counter
        off = _EXACT_OFFSET + item
        v = buf[off]
        if v < 255:
            buf[off] = v + 1
    else:
        # Count-Min Sketch
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            off = _SKETCH_OFFSET + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(buf, off)
            if v < 0xFFFF:
                _write_u16(buf, off, v + 1)
            
    return bytes(buf)

def query(state, item):
    if item < _EXACT_SIZE:
        off = _EXACT_OFFSET + item
        return state[off]
    else:
        mn = 0xFFFF
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            off = _SKETCH_OFFSET + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(state, off)
            if v < mn:
                mn = v
        return mn
