import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Build via Kronecker-product-style construction:
    # The multiplication tensor M_{n,m,p} = M_{n,m,1} ⊗ I_p (in a suitable sense)
    # For each column k of B, we need the full M_{n,m,1} tensor.
    #
    # Rank column r corresponds to (k, i, j) where:
    #   r = k * (n*m) + i * m + j
    #
    # U[r] has a 1 at position (i, j) -> index i*m + j
    # V[r] has a 1 at position (j, k) -> index j*p + k
    # W[r] has a 1 at position (k, i) -> index k*n + i  (trace convention)

    r = np.arange(rank)
    k = r // (n * m)
    i = (r % (n * m)) // m
    j = r % m

    u_pos = i * m + j
    v_pos = j * p + k
    w_pos = k * n + i

    U[u_pos, r] = 1
    V[v_pos, r] = 1
    W[w_pos, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
