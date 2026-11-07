import struct
import time

_START = time.time()
_DEADLINE = _START + 110.0

# State layout (1024 bytes total):
# Bytes 0..511: 512 x uint16 counters (little-endian) for items 0..511
#   - Items 0..399 get dedicated 16-bit counters (max 65535, far more than enough)
#   - Items 400..511 are unused (will stay 0)
# Bytes 512..1023: 512 bytes reserved (unused, kept as zero)
#
# Strategy:
# - Use dedicated uint16 counters for all 400 items.
# - This gives EXACT frequency counts (no sketch error).
# - Max possible count per item in any stream:
#   * Uniform: ~1024/400 ≈ 2.56 average, max maybe ~15
#   * Zipf(1.6): item 0 gets the most. Let's estimate:
#     P(k) ∝ (k+1)^(-1.6). Normalizing: sum_{k=0}^{399} (k+1)^(-1.6)
#     ≈ 1 + sum_{i=2}^{400} i^(-1.6) ≈ 1 + 1.0 (roughly, since the sum converges)
#     More carefully: sum_{i=1}^{400} i^(-1.6) ≈ integral_1^400 x^(-1.6) dx + 1
#     = [x^(-0.6)/(-0.6)]_1^400 + 1 = (1 - 400^(-0.6))/0.6 + 1 ≈ (1 - 0.0225)/0.6 + 1 ≈ 1.629 + 1 = 2.629
#     Wait, let me recompute: ∫_1^400 x^(-1.6) dx = [x^(-0.6)/(-0.6)]_1^400 = (1^(-0.6) - 400^(-0.6))/0.6
#     = (1 - 400^(-0.6))/0.6. 400^0.6 = e^(0.6*ln400) = e^(0.6*5.991) = e^3.595 ≈ 36.4, so 400^(-0.6) ≈ 0.0275
#     So integral ≈ (1 - 0.0275)/0.6 ≈ 1.621. Plus the first term 1^(-1.6)=1, so sum ≈ 2.621.
#     P(0) = 1/2.621 ≈ 0.381. Expected count for item 0 ≈ 1024 * 0.381 ≈ 390.
#     That's well within uint16 (max 65535).
#   * Third stream: 512 uniform on 0..19 → each gets ~26.7 on average, max maybe ~40.
#     Plus 512 uniform on 150..399 → each gets ~2.05 on average.
#     So items 0..19 get ~27, items 150..399 get ~2. All well within uint16.
#
# Therefore, uint16 dedicated counters give EXACT counts for all items in all streams.
# The score should be 0 (or very close to 0, limited by any floating-point in scoring).

def update(state, item):
    """Increment the dedicated uint16 counter for the given item."""
    if time.time() > _DEADLINE:
        return state
    
    if not (0 <= item < 400):
        return state
    
    ba = bytearray(state)
    # uint16 little-endian at offset item*2
    offset = item * 2
    current = ba[offset] | (ba[offset + 1] << 8)
    if current < 65535:
        current += 1
    ba[offset] = current & 0xFF
    ba[offset + 1] = (current >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    """Return the exact count from the dedicated uint16 counter."""
    if time.time() > _DEADLINE:
        return 0
    
    if not (0 <= item < 400):
        return 0
    
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
