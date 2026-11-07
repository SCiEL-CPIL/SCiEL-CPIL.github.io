def _halton(index, base):
    result = 0
    f = 1
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

def _halton_sequence(n, dims):
    points = np.zeros((n, dims))
    for d in range(dims):
        base = d + 2 # 2, 3, 5...
        for i in range(n):
            points[i, d] = _halton(i + 1, base) # +1 to skip 0
    return points
