def construct():
    # Use additive coloring modulo 17
    # This is guaranteed to be valid for n=16
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
