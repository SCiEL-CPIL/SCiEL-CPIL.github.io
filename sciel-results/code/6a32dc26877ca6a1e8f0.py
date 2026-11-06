"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin under 90s limit
    p = 1009
    
    # Precompute root lookup: root[r] = smallest x in [1, p-1] with x^2 mod p == r
    root = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if root[r] == 0:
            root[r] = x
    
    # Build bitset: bit i set iff i is a non-zero quadratic residue mod p
    bitset = 0
    for x in range(1, p):
        r = (x * x) % p
        bitset |= (1 << r)
    
    # Find the smallest s in [1, 1004] such that s..s+4 are all non-zero QRs
    # Using bitwise check: (bitset >> s) & 0b11111 == 0b11111
    mask = 0b11111
    for s in range(1, 1005):
        if time.time() > deadline:
            raise RuntimeError('Deadline exceeded')
        if (bitset >> s) & mask == mask:
            # Verify roots exist and are in valid range [1, 1008]
            roots = []
            valid = True
            for i in range(5):
                r = s + i
                x = root[r]
                if x == 0 or x < 1 or x > 1008:
                    valid = False
                    break
                roots.append(x)
            if valid:
                return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
