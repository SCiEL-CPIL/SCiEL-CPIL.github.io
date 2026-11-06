def priority(vector, n):
    # FNV-1a hash
    h = 14695981039346656037
    for x in vector:
        h ^= x
        h = (h * 1099511628211) % (2**64)
    return float(h)
