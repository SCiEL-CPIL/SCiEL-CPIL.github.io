import struct

NUM_ROWS = 8
NUM_COLS = 64
STATE_SIZE = 1024

# Primes for hashing
# We need 8 distinct hash functions.
# Using a standard set of large primes.
PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    1274126177, 15485863, 15485863, 15485863 # Need to pick better ones
]
# Let's pick 8 distinct large primes.
# 1035493289, 775374203, 1495862053, 2015145641 are from the prompt.
# Let's add: 1103515245, 1234567891, 1234567891, 1234567891 -- no, distinct.
# 1000003, 1000033, 1000037, 1000039 are small.
# Let's use:
# 1035493289
# 775374203
# 1495862053
# 2015145641
# 2654435761
# 3221225473
# 3865470565
# 4059307693
# Salts can be 31 * row as in the prompt, or distinct constants.
# The prompt used: index = ((item * p + 31 * row) % 2147483647) % 64
# We will use similar logic with 8 rows.

SALTS = [31, 61, 97, 131, 167, 199, 227, 257] # Distinct salts
MOD_LARGE = 2147483647

def update(state, item):
    # Convert state to array of unsigned shorts
    # state is bytes
    # We use struct.unpack to get 512 unsigned shorts
    # '<512H'
    counters = struct.unpack('<512H', state)
    
    # Convert to list for mutable modification
    # list(counters) is fast enough
    c = list(counters)
    
    for r in range(NUM_ROWS):
        p = PRIMES[r]
        s = SALTS[r]
        # Hash calculation
        idx = ((item * p + s) % MOD_LARGE) % NUM_COLS
        pos = r * NUM_COLS + idx
        c[pos] += 1
        # No overflow check needed as max count < 65535
    
    # Pack back
    return struct.pack('<512H', *c)

def query(state, item):
    counters = struct.unpack('<512H', state)
    
    min_val = 65535
    for r in range(NUM_ROWS):
        p = PRIMES[r]
        s = SALTS[r]
        idx = ((item * p + s) % MOD_LARGE) % NUM_COLS
        pos = r * NUM_COLS + idx
        val = counters[pos]
        if val < min_val:
            min_val = val
            
    return min_val
