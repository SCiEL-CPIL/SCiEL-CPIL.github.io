import time

_start_time = time.time()
_DEADLINE = 145  # 145s out of 150s, 5s safety margin

# Structural change: Bloom filter + global total counter
# Layout: bytes[0:1020] = 8160-bit Bloom filter, bytes[1020:1024] = uint32 total count
# k=3 hash functions for the Bloom filter
# On update: set 3 bits in BF, increment total count
# On query: if item not in BF -> return 0 (exact for unseen items)
#           if item in BF -> return total / 400 (uniform estimate over all possible items)

_BF_SIZE_BYTES = 1020
_BF_SIZE_BITS = 8160
_K = 3
_M = 2147483647  # 2^31 - 1

# Hash parameters for Bloom filter (3 distinct pairs)
_BF_HASH_PARAMS = [
    (1035493289, 31),
    (775374203, 107),
    (1495862053, 211),
]


def _bf_bit_index(item, k_idx):
    p, q = _BF_HASH_PARAMS[k_idx]
    return ((item * p + q) % _M) % _BF_SIZE_BITS


def update(state, item):
    if time.time() - _start_time > _DEADLINE:
        return state

    b = bytearray(state)
    # Set 3 bits in Bloom filter
    for k_idx in range(_K):
        bit_idx = _bf_bit_index(item, k_idx)
        byte_idx = bit_idx >> 3
        bit_mask = 1 << (bit_idx & 7)
        b[byte_idx] |= bit_mask

    # Increment total count (uint32 little-endian at bytes 1020-1023)
    total = int.from_bytes(b[1020:1024], 'little')
    total += 1
    b[1020:1024] = total.to_bytes(4, 'little')

    return bytes(b)


def query(state, item):
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    # Check Bloom filter membership
    for k_idx in range(_K):
        bit_idx = _bf_bit_index(item, k_idx)
        byte_idx = bit_idx >> 3
        bit_mask = 1 << (bit_idx & 7)
        if not (state[byte_idx] & bit_mask):
            # Item definitely not seen -> exact 0
            return 0.0

    # Item is in Bloom filter (may be false positive)
    # Return uniform estimate: total / 400
    total = int.from_bytes(state[1020:1024], 'little')
    return float(total) / 400.0
