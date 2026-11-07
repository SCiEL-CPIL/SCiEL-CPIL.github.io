import struct

# EVOLVE-BLOCK-START
PRIMES = [1035493289, 775374203, 1495862053, 2015145641]
WIDTH = 78
DEPTH = 4
SKETCH_OFFSET = 400 # Start of sketch area
COUNTER_SIZE = 2

def update(state, item):
    b = bytearray(state)
    val = b[item]
    if val < 255:
        val += 1
        b[item] = val
    else:
        # Overflow: update sketch
        for i in range(DEPTH):
            p = PRIMES[i]
            idx = ((item * p + 31 * i) % 2147483647) % WIDTH
            offset = SKETCH_OFFSET + i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
            v = struct.unpack_from('<H', b, offset)[0]
            v += 1
            if v > 65535: v = 65535
            struct.pack_into('<H', b, offset, v)
    return bytes(b)

def query(state, item):
    val = state[item]
    if val < 255:
        return val
    else:
        min_val = 65535
        for i in range(DEPTH):
            p = PRIMES[i]
            idx = ((item * p + 31 * i) % 2147483647) % WIDTH
            offset = SKETCH_OFFSET + i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
            v = struct.unpack_from('<H', state, offset)[0]
            if v < min_val:
                min_val = v
        return 255 + min_val
# EVOLVE-BLOCK-END
