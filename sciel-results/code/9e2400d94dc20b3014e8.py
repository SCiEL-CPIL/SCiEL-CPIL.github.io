def _metric(x):
    C = np.correlate(x, x, 'full')
    n = len(x)
    peak = C[n-1]  # C[0]
    offpeak = np.max(np.abs(np.concatenate([C[:n-1], C[n:]])))
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return offpeak / (s*s)
