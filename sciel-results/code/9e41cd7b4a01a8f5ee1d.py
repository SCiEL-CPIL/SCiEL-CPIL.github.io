import struct

# 4 rows, 128 counters per row, uint16
# Total: 4 * 128 * 2 = 1024 bytes

# Hash parameters
# We need 4 distinct hash functions.
# h_i(x) = (a_i * x + b_i) % 128
# We can use large primes for a_i and distinct b_i.
# Or use a simple universal hash scheme.

A = [1035493289, 775374203, 1495862053, 2015145641]
B = [31, 57, 91, 113] # Arbitrary offsets

def _get_indices(item):
    indices = []
    for i in range(4):
        h = (item * A[i] + B[i]) % 128
        indices.append(i * 128 + h)
    return indices

def update(state, item):
    # Unpack 512 uint16s
    counters = list(struct.unpack('<512H', state))
    indices = _get_indices(item)
    for idx in indices:
        counters[idx] += 1
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = list(struct.unpack('<512H', state))
    indices = _get_indices(item)
    min_val = min(counters[idx] for idx in indices)
    return float(min_val)
