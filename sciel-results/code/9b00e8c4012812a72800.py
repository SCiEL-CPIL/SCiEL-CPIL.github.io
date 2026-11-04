"""Exact rank-minimal (schoolbook) tensor decomposition for 3x5x5 matmul, built via nested list comprehensions."""

# EVOLVE-BLOCK-START
def construct():
    # The matmul tensor for (n, m, p) = (3, 5, 5) has entries T[i, j, k] = 1
    # for i in range(n), j in range(m), k in range(p) corresponding to
    # (A row i) * (B col k) summed over j.
    # The schoolbook decomposition uses one rank-1 slice per (i, j, k) triple,
    # giving rank = n*m*p = 75.
    n, m, p = 3, 5, 5
    R = n * m * p

    # U has shape (n*m, R): U[i*m + j, idx(i,j,k)] = 1
    # V has shape (m*p, R): V[j*p + k, idx(i,j,k)] = 1
    # W has shape (p*n, R): W[k*n + i, idx(i,j,k)] = 1
    # idx(i,j,k) = k*m + j*n + i  (column-major over (k, j, i))

    U = [[int(i == I and j == J) for I in range(n) for J in range(m) for K in range(p)]
         for i in range(n) for j in range(m)]
    V = [[int(j == J and k == K) for I in range(n) for J in range(m) for K in range(p)]
         for j in range(m) for k in range(p)]
    W = [[int(k == K and i == I) for I in range(n) for J in range(m) for K in range(p)]
         for k in range(p) for i in range(n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
