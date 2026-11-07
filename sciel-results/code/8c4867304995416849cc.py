import struct

def update(state, item):
    state = bytearray(state)
    if item < 128:
        # Exact count
        offset = item * 4
        val = struct.unpack_from('<I', state, offset)[0]
        val += 1
        struct.pack_into('<I', state, offset, val)
    else:
        # Sketch count
        # 4 rows, 32 counters each
        # Hash functions
        primes = (1035493289, 775374203, 1495862053, 2015145641)
        base_offset = 512
        for r in range(4):
            idx = ((item * primes[r] + 31 * r) % 2147483647) % 32
            offset = base_offset + r * 32 * 4 + idx * 4
            val = struct.unpack_from('<I', state, offset)[0]
            val += 1
            struct.pack_into('<I', state, offset, val)
    return bytes(state)

def query(state, item):
    if item < 128:
        offset = item * 4
        return struct.unpack_from('<I', state, offset)[0]
    else:
        primes = (1035493289, 775374203, 1495862053, 2015145641)
        base_offset = 512
        min_val = float('inf')
        for r in range(4):
            idx = ((item * primes[r] + 31 * r) % 2147483647) % 32
            offset = base_offset + r * 32 * 4 + idx * 4
            val = struct.unpack_from('<I', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val
