def _construct_circulant(k, g, n, deadline):
    """Try circulant graphs C_n(1, a, -1, -a) for k=4."""
    if k != 4:
        return None
        
    for a in range(2, n // 2 + 1):
        if time.time() > deadline:
            break
        edges = []
        for v in range(n):
            for step in [1, -1, a, -a]:
                u = (v + step) % n
                if v < u:
                    edges.append((v, u))
        if _verify_girth(n, edges, k, g):
            return n, edges
    return None
