for u in range(p):
    for s in S:
        v = (u + s) % p
        if u < v:
            edges.append((u, v))
