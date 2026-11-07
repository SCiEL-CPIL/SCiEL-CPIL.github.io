import struct

def update(state, item):
    # Exact frequency storage: 400 uint8 counters in the first 400 bytes.
    # Since each stream has 1024 items over 400 possible values,
    # no individual item can appear more than 1024 times, but actually
    # the maximum possible count for any single item is bounded by the
    # stream length. However, with 1024 items and 400 values, the max
    # count for any item is at most 1024, which exceeds 255.
    # 
    # Wait - let me reconsider. The task says "Three insertion-only streams 
    # of 1024 items". Each stream is processed independently with a fresh 
    # interpreter per 256-item chunk. The state persists across chunks.
    # So for a single stream of 1024 items, an item could theoretically 
    # appear up to 1024 times (if all items were the same), but in practice
    # with uniform/Zipf distributions, the max count will be much lower.
    #
    # For uniform on 0..399 with 1024 items: expected count per item ~2.56,
    # max likely around 10-15.
    # For Zipf(1.6): item 0 could have high frequency. Let's calculate:
    #   Zipf with s=1.6: P(x) = x^(-1.6) / H(400, 1.6)
    #   H(400, 1.6) ≈ sum_{k=1}^{400} k^(-1.6) ≈ 1 + 2^(-1.6) + ... 
    #   This is approximately 1 + 0.33 + 0.159 + ... ≈ 2.5 (rough estimate)
    #   So P(0) ≈ 1/2.5 ≈ 0.4, meaning item 0 appears ~409 times.
    #   That exceeds 255!
    #
    # So uint8 won't work for all items. Let me use a hybrid approach:
    # - Use uint16 for all 400 items: 400 * 2 = 800 bytes. Fits in 1024.
    # - Or use uint32 for all 400 items: 400 * 4 = 1600 bytes. Too much.
    # - Use uint16: max value 65535, which is way more than enough.
    
    buf = bytearray(state)
    
    # Each item gets 2 bytes (uint16) at offset item * 2
    offset = item * 2
    val = buf[offset] | (buf[offset + 1] << 8)
    val += 1
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    # Read the uint16 counter for this item
    offset = item * 2
    val = state[offset] | (state[offset + 1] << 8)
    return val
