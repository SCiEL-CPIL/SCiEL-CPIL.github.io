def _poly_mul(p, q):
    # p and q are 1D arrays
    # We want coefficients up to len(p)-1
    # np.convolve(p, q) gives length len(p)+len(q)-1
    # We take the first len(p)
    return np.convolve(p, q)[:len(p)]
