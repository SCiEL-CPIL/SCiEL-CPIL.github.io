import struct

# Two-tier exact counter design:
# - Top 128 items (0..127) get 16-bit counters in bytes 0..255
# - Items 128..399 get 8-bit counters in bytes 256..639
# - Remaining bytes 640..1023 are unused (zero)
#
# This eliminates all collision/sketch noise for the high-frequency core.
# For Zipf(1.6), the top items have very high counts; exact tracking is critical.
# For the bimodal stream, items 0..19 (left side) are in the top-128 tier.
# Items 150..399 (right side) are in the 8-bit tier; max count per item is 512,
# which fits in uint8 (0..255)? No, 512 > 255. 
# Wait: 512 uniform on 150..399 means each item in 150..399 gets ~512/250 ≈ 2.05 on average.
# Actually, 512 items uniformly distributed over 250 values (150..399 inclusive is 250 values).
# So max count for any single item is roughly 512/250 * some factor. With uniform, max is small.
# Actually with 512 items over 250 slots, expected is ~2, max maybe 5-10. Fits in uint8 easily.
#
# For Zipf(1.6) mod 400: the top items get very high counts. Item 0 could get ~100+ counts.
# 16-bit counters handle up to 65535, so no overflow.
#
# For uniform on 0..399: each item gets ~1024/400 ≈ 2.56 on average. Max maybe 5-8. Fits in uint8.

def update(state, item):
    buf = bytearray(state)
    if item < 128:
        # 16-bit counter at offset item * 2
        off = item * 2
        cur = buf[off] | (buf[off + 1] << 8)
        if cur < 65535:
            cur += 1
        buf[off] = cur & 0xFF
        buf[off + 1] = (cur >> 8) & 0xFF
    else:
        # 8-bit counter at offset 256 + (item - 128)
        off = 256 + (item - 128)
        if buf[off] < 255:
            buf[off] += 1
    return bytes(buf)

def query(state, item):
    if item < 128:
        off = item * 2
        return state[off] | (state[off + 1] << 8)
    else:
        off = 256 + (item - 128)
        return state[off]
