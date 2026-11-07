import struct

# EVOLVE-BLOCK-START
# We use 8 rows of 64 columns, each counter is 16-bit unsigned int.
# Total counters: 512.
# Total bytes: 512 * 2 = 1024.

# Constants for hashing - using large odd constants for good distribution
P0 = 1035493289
P1 = 775374203
P2 = 1495862053
P3 = 2015145641
P4 = 1234567891
P5 = 987654323
P6 = 314159265
P7 = 271828183

MOD = 2147483647  # 2^31 - 1

def update(state, item):
    s = bytearray(state)
    
    # Row 0
    idx = ((item * P0 + 1) % MOD) % 64
    pos = idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 1
    idx = ((item * P1 + 2) % MOD) % 64
    pos = 64 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 2
    idx = ((item * P2 + 3) % MOD) % 64
    pos = 128 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 3
    idx = ((item * P3 + 4) % MOD) % 64
    pos = 192 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 4
    idx = ((item * P4 + 5) % MOD) % 64
    pos = 256 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 5
    idx = ((item * P5 + 6) % MOD) % 64
    pos = 320 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 6
    idx = ((item * P6 + 7) % MOD) % 64
    pos = 384 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    # Row 7
    idx = ((item * P7 + 8) % MOD) % 64
    pos = 448 + idx
    val = s[pos] | (s[pos+1] << 8)
    val += 1
    s[pos] = val & 0xFF
    s[pos+1] = (val >> 8) & 0xFF
    
    return bytes(s)

def query(state, item):
    min_val = 1024  # Max possible count
    
    # Row 0
    idx = ((item * P0 + 1) % MOD) % 64
    pos = idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 1
    idx = ((item * P1 + 2) % MOD) % 64
    pos = 64 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 2
    idx = ((item * P2 + 3) % MOD) % 64
    pos = 128 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 3
    idx = ((item * P3 + 4) % MOD) % 64
    pos = 192 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 4
    idx = ((item * P4 + 5) % MOD) % 64
    pos = 256 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 5
    idx = ((item * P5 + 6) % MOD) % 64
    pos = 320 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 6
    idx = ((item * P6 + 7) % MOD) % 64
    pos = 384 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    # Row 7
    idx = ((item * P7 + 8) % MOD) % 64
    pos = 448 + idx
    val = state[pos] | (state[pos+1] << 8)
    if val < min_val: min_val = val
    
    return min_val
# EVOLVE-BLOCK-END
