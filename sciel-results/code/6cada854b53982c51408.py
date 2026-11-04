def construct(k, g):
    q = k
    # d determines girth; girth >= 2*floor(d/2)+1 for good c sequence
    # For girth g, we need d >= g-1
    d = max(2, g - 1)
    n = 2 * q ** d
    # Limit size
    while n > 200000 and d > 2:
        d -= 1
        n = 2 * q ** d
    
    # c sequence: c_i for i=2..d
    # For maximum girth, use c_i = 1
    c = {i: 1 for i in range(2, d+1)}
    
    # Vertices indexed by (side, x1, x2, ..., xd)
    # side in {0,1}, xi in {0,...,q-1}
    # Index: side * q^d + x1*q^(d-1) + x2*q^(d-2) + ... + xd
    
    edges = []
    edge_set = set()
    
    for side in [0, 1]:
        for x1 in range(q):
            for x2 in range(q):
                ... # nested loops for d dimensions - too slow for large d
    
    # Use itertools.product
    from itertools import product
    for side in [0, 1]:
        for coords in product(range(q), repeat=d):
            v_idx = side * (q ** d) + sum(coords[i] * (q ** (d-1-i)) for i in range(d))
            # Neighbors: (1-side, y) where y is computed
            for a in range(q):
                # y1 = a
                # y_i = coords[i-1] + (1 if side==0 else -1) * c[i] * a for i>=2
                y = list(coords)
                y[0] = a
                for i in range(1, d):
                    y[i] = (coords[i] + (1 if side == 0 else -1) * c[i+1] * a) % q
                u_idx = (1 - side) * (q ** d) + sum(y[i] * (q ** (d-1-i)) for i in range(d))
                e = (min(v_idx, u_idx), max(v_idx, u_idx))
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)
    
    return n, edges
