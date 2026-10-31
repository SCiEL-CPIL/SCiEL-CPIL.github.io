def priority(vector, n):
    # Fast 64-bit hash
    h = 0x9E3779B97F4A7C15
    for x in vector:
        h ^= x
        h = (h << 13) | (h >> 51)
        h *= 0x517CC1B727220A95
        h &= 0xFFFFFFFFFFFFFFFF
    return float(h)
