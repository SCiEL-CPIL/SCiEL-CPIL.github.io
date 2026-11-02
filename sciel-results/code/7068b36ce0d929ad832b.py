<<<<<<< SEARCH
def _lift(sigma, tau, m):
    n = _BASE_V * m
    adj = [[] for _ in range(n)]
    for (u, v) in _BASE_EDGES:
        e = _EDGE_IDX[(u, v)]
        s = sigma[e] % m
        t = tau[e]
        for f in range(m):
            a = u * m + f
            b = v * m + (f + s) % m
            adj[a].append(b)
            adj[b].append(a)
            if t:
                a2 = u * m + (f + m // 2) % m
                b2 = v * m + (f + s + m // 2) % m
                adj[a2].append(b2)
                adj[b2].append(a2)
    return n, adj
=======
def _lift(sigma, m):
    """Lift K_{5,5} by Z_m using voltage assignment sigma."""
    n = _BASE_V * m
    adj = [[] for _ in range(n)]
    for (u, v) in _BASE_EDGES:
        e = _EDGE_IDX[(u, v)]
        s = sigma[e] % m
        for f in range(m):
            a = u * m + f
            b = v * m + (f + s) % m
            adj[a].append(b)
            adj[b].append(a)
    return n, adj
>>>>>>> REPLACE

