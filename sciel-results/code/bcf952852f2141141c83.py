def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))
