    a = state[:, i]
    b = state[:, j]
    m = np.minimum(a, b)
    M = np.maximum(a, b)
    state[:, i] = m
    state[:, j] = M
    