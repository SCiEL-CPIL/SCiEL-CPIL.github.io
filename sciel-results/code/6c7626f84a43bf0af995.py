def instance(n, p, seed):
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
