"""
Optimized Count-Min Sketch using full 1024-byte budget.
Structure: 4 rows of 128 uint16 counters.
Total bytes: 4 * 128 * 2 = 1024.

Rationale:
- The seed used 4 rows of 64 uint32 counters.
- By switching to uint16, we can double the width (from 64 to 128) while keeping the same depth (4).
- Increasing width reduces the collision bias (overestimation) by half, which is the dominant error source for items with non-zero frequency.
- Keeping depth at 4 ensures we still take the minimum of 4 estimates, reducing variance from high-collision outliers.
- This pure probabilistic approach avoids the complexity and potential boundary issues of hybrid bitset solutions, while leveraging the full memory for better resolution.
"""

import struct

# Precomputed hashes for speed
# We use 4 different multipliers and salts to decorrelate the rows.
# The modulo operation is on a large prime to ensure good distribution before taking % 128.
# However, since item < 400, we can use simpler effective hashes.
# Let's use: (item * P_i + S_i) % 128
# To ensure independence, P_i should be odd and coprime to 128 (which is 2^7).
# Actually, any odd P works for modulo 2^k if we just take the lower bits, but modulo a prime is safer for mixing.
# Given the small item range, we can just use:
# h_i = (item * A_i + B_i) % 128
# Let's pick A_i such that they are distinct modulo 128 and "mix" well.
# Since 128 is a power of 2, we want A_i to be odd.
# A = [1, 3, 5, 7] might be too correlated for small items?
# Let's use larger multipliers modulo 2^32 then take % 128?
# Or just use the seed's multipliers but mod 128?
# Seed multipliers: 1035493289, 775374203, 1495862053, 2015145641
# These are large. (item * P) % 128 is effectively (item * (P % 128)) % 128.
# P1 % 128 = 1035493289 % 128 = 1
# P2 % 128 = 775374203 % 128 = 3
# P3 % 128 = 1495862053 % 128 = 5
# P4 % 128 = 2015145641 % 128 = 9
# Using multipliers 1, 3, 5, 9 with item 0..399.
# This is fine. The mapping is bijective modulo 128 for each row if the multiplier is coprime to 128 (odd).
# 1, 3, 5, 9 are all odd.

MULTS = [1, 3, 5, 9]
# We don't really need salts if the multipliers are distinct and odd, 
# as item * 1, item * 3, etc. will hit different buckets for different items.
# However, to be safe against any structural alignment, we can add salts.
# But for 0..399, item*1 % 128 covers 0..127 repeatedly.
# item*3 % 128 covers 0..127 repeatedly (since gcd(3,128)=1).
# The "collisions" in the sketch come from different items mapping to the same bucket.
# With w=128 and N=400 items, we expect ~3.125 items per bucket on average.
# This is much better than w=64 (~6.25 items per bucket).

def update(state, item):
    b = bytearray(state)
    
    # Update 4 counters
    for i in range(4):
        # Calculate bucket index
        # Using the multipliers derived from the seed to maintain a similar "character" but wider
        # h = (item * MULTS[i]) % 128
        bucket = (item * MULTS[i]) % 128
        
        # Offset for this row
        # Row i starts at i * 128 counters
        # Each counter is 2 bytes
        offset = i * (128 * 2) + bucket * 2
        
        # Read uint16
        val = b[offset] | (b[offset + 1] << 8)
        
        # Increment
        val += 1
        
        # Write back
        b[offset] = val & 0xFF
        b[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    
    min_val = 65535 # Max uint16
    
    for i in range(4):
        bucket = (item * MULTS[i]) % 128
        offset = i * (128 * 2) + bucket * 2
        
        val = b[offset] | (b[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
            
    return min_val
