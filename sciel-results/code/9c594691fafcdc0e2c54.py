"""
Dual-structure adaptive frequency sketch.
State layout (1024 bytes):
- Bytes 0-255: 256 dedicated uint8 counters for items 0-199.
- Bytes 256-1023: 5 rows x 128 columns of uint8 counters for a Count-Min sketch.
Items 0-199 are counted exactly in the dedicated section (safe because max count per item in any stream is 512 < 255? No, 512 > 255. 
Wait, let's re-evaluate capacity.
Stream 3: 512 uniform on 0..19. Max count ~ 512/20 = 25.6. 
Stream 1: Uniform 0..399. Max count ~ 1024/400 = 2.56.
Stream 2: Zipf(1.6) mod 400. Item 0 has highest prob. P(0) ~ 1/1.6^0 / H(400,1.6). H ~ ln(400) + gamma. ~ 6.4. P(0) ~ 1/6.4 ~ 0.15. Count ~ 150.
So uint8 (max 255) is sufficient for all items in 0..199 across all streams.
Items 200..399 go to the CM sketch.
"""
# EVOLVE-BLOCK-START

_NUM_DEDICATED = 256
_NUM_CM_ROWS = 5
_NUM_CM_COLS = 128
_CM_OFFSET = _NUM_DEDICATED

# Primes for the CM sketch part
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    
    if item < _NUM_DEDICATED:
        # Increment dedicated counter
        idx = item
        val = buf[idx]
        if val < 255:
            val += 1
        buf[idx] = val
    else:
        # Increment CM sketch counters
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            # Hash to 128 columns
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = buf[offset]
            if val < 255:
                val += 1
            buf[offset] = val
            
    return bytes(buf)


def query(state, item):
    if item < _NUM_DEDICATED:
        # Return exact count
        return state[item]
    else:
        # Return min of CM sketch rows
        best = 255
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = state[offset]
            if val < best:
                best = val
        return best

# EVOLVE-BLOCK-END
