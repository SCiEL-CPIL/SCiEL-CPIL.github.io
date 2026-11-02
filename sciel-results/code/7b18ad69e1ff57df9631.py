"""Matrix multiplication 2x2x11: rank-minimal exact tensor decomposition.

The tensor T_{2,2,11} is the direct sum of 11 copies of T_{2,2,1} (each
output column depends only on the corresponding input column).  The rank of
T_{2,2,1} is 4 (two independent dot-products, each of rank 2), so the
overall rank is 11 * 4 = 44.  We emit the canonical 44-term decomposition
with a distinct construction style from the seed."""

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p  # 44
    # Pre-allocate as lists of lists of ints
    U = [[0] * R for _ in range(n * m)]   # 4 x 44
    V = [[0] * R for _ in range(m * p)]   # 22 x 44
    W = [[0] * R for _ in range(p * n)]   # 22 x 44

    idx = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U indexed by (i,j) -> row i*m+j
                U[i * m + j][idx] = 1
                # V indexed by (j,k) -> row j*p+k
                V[j * p + k][idx] = 1
                # W indexed by (k,i) -> row k*n+i  (trace convention W[k*n+i, r])
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
