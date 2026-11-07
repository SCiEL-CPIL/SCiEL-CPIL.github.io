"""Exact frequency table: 400 bytes of 1-byte counters within 1024-byte state."""

def update(state, item):
    """Increment the exact count for 'item' (0..399) stored at byte index 'item'."""
    # state is a bytes object of length <= 1024
    # Convert to bytearray, increment the byte at position 'item', return as bytes
    ba = bytearray(state)
    # Ensure length is at least 400 (should already be 1024 from init)
    if len(ba) < 400:
        ba.extend(b'\x00' * (400 - len(ba)))
    # item is in 0..399, so ba[item] is the counter for that item
    # Max count per item: in the worst stream (Zipf), a single item could appear many times
    # but with 1024 items total and 400 possible values, max count is bounded well under 255
    # Actually, for Zipf(1.6) on 400 items, the most frequent item gets roughly 1024 * (1 / sum_{i=1}^{400} i^{-1.6})
    # sum_{i=1}^{400} i^{-1.6} ≈ integral from 1 to 400 of x^{-1.6} dx = [x^{-0.6}/(-0.6)]_1^400 = (1 - 400^{-0.6})/0.6
    # 400^{-0.6} = (400)^{-0.6} ≈ exp(-0.6 * ln(400)) = exp(-0.6 * 5.99) ≈ exp(-3.594) ≈ 0.0275
    # So sum ≈ (1 - 0.0275)/0.6 ≈ 1.62
    # Most frequent count ≈ 1024 / 1.62 ≈ 632... wait that's over 255!
    
    # Let me recalculate. Zipf distribution: P(i) ∝ i^{-1.6} for i = 1, 2, ..., 400
    # Then modulo 400 maps to 0..399.
    # The probability of item 1 (which maps to 1 % 400 = 1) is proportional to 1^{-1.6} = 1
    # Total mass = sum_{i=1}^{400} i^{-1.6}
    # P(1) = 1 / sum_{i=1}^{400} i^{-1.6}
    # Expected count of item 1 = 1024 * P(1) = 1024 / sum_{i=1}^{400} i^{-1.6}
    
    # Let me compute sum_{i=1}^{400} i^{-1.6} more carefully
    # This is the generalized harmonic number H(400, 1.6)
    # H(n, s) ≈ n^{1-s}/(1-s) + zeta(s) + 1/(2n^s) + ... for s > 1
    # For s = 1.6, 1-s = -0.6
    # H(400, 1.6) ≈ 400^{-0.6}/(-0.6) + zeta(1.6) + 1/(2*400^{1.6})
    # 400^{-0.6} = exp(-0.6 * ln(400)) = exp(-0.6 * 5.9915) = exp(-3.5949) ≈ 0.02745
    # 400^{-0.6}/(-0.6) = 0.02745 / (-0.6) = -0.04575... 
    
    # Wait, the formula is H(n,s) ≈ n^{1-s}/(1-s) + zeta(s) + ...
    # n^{1-s} = 400^{-0.6} ≈ 0.02745
    # n^{1-s}/(1-s) = 0.02745 / (-0.6) ≈ -0.04575
    # zeta(1.6) ≈ 2.06 (approximately)
    # So H(400, 1.6) ≈ -0.04575 + 2.06 ≈ 2.014
    # Hmm, but this approximation isn't great for small n relative to the convergence.
    
    # Let me just estimate: the sum is dominated by the first few terms.
    # 1^{-1.6} = 1
    # 2^{-1.6} ≈ 0.3299
    # 3^{-1.6} ≈ 0.1831
    # 4^{-1.6} ≈ 0.1250
    # 5^{-1.6} ≈ 0.0918
    # 6^{-1.6} ≈ 0.0699
    # 7^{-1.6} ≈ 0.0551
    # 8^{-1.6} ≈ 0.0447
    # 9^{-1.6} ≈ 0.0369
    # 10^{-1.6} ≈ 0.0316
    # Sum of first 10 ≈ 1 + 0.3299 + 0.1831 + 0.1250 + 0.0918 + 0.0699 + 0.0551 + 0.0447 + 0.0369 + 0.0316 ≈ 2.068
    # The rest from 11 to 400 adds maybe 0.3-0.5 more
    # So total sum ≈ 2.4-2.6
    # P(1) ≈ 1/2.5 ≈ 0.4
    # Expected count of item 1 ≈ 1024 * 0.4 ≈ 409.6
    
    # That's over 255! So 1 byte is NOT sufficient for exact counts in the Zipf case.
    
    # I need to use 2 bytes per item, or use a different approach.
    # 400 items * 2 bytes = 800 bytes, which fits in 1024.
    
    # Let's use 2 bytes (little-endian uint16) per item. Max value 65535, well above any possible count.
    
    ba = bytearray(state)
    if len(ba) < 800:
        ba.extend(b'\x00' * (800 - len(ba)))
    
    offset = item * 2
    current = ba[offset] | (ba[offset + 1] << 8)
    new_val = current + 1
    ba[offset] = new_val & 0xFF
    ba[offset + 1] = (new_val >> 8) & 0xFF
    
    return bytes(ba)


def query(state, item):
    """Return the exact frequency estimate for 'item' (0..399)."""
    if len(state) < 800:
        return 0
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
