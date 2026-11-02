def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # ...
                s += U[u][c] * V[v][c] * W[w][c]
