"""Matrix multiplication 2x2 x 2x7: rank-16 decomposition via column-wise 2x2 products."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    # U shape: (n*m, rank) = (4, 16)
    # V shape: (m*p, rank) = (14, 16)
    # W shape: (p*n, rank) = (14, 16)
    #
    # Convention: C[k*n+i, r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each column k of C (k in 0..6), C[:,k] = A * B[:,k] (2x2 * 2x1).
    # The 2x2 x 2x1 product has rank 4 with scalar multiplications:
    #   t0 = A[0,0]*B[0,k]  -> contributes to C[0,k]
    #   t1 = A[0,1]*B[1,k]  -> contributes to C[0,k]
    #   t2 = A[1,0]*B[0,k]  -> contributes to C[1,k]
    #   t3 = A[1,1]*B[1,k]  -> contributes to C[1,k]
    #
    # Total rank = 7 columns * 4 = 28 ... but we can do better?
    # Actually each column is independent, so rank = 7*4 = 28.
    #
    # Wait - re-examining: the tensor for C[i,k] = sum_j A[i,j]*B[j,k]
    # The "A" matrix is 2x2 (4 entries), B is 2x7 (14 entries), C is 2x7 (14 entries).
    # Tensor T_{i,j,k} for fixed (i,k) sums over j. This is exactly 7 independent
    # copies of the 2x1 vector-matrix product (A times a column of B).
    #
    # Each 2x2 * 2x1 has rank 4 (it's a 2x2x1 tensor, rank = 4 = 2*2*1).
    # 7 independent copies -> rank 28.
    #
    # Hmm, but can Strassen reduce the per-column rank? No - 2x2*2x1 is just
    # 4 independent products (each C[i] uses A[i,0]*B[0] and A[i,1]*B[1],
    # and different i use different A rows). Actually let me check:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]  (2 multiplications)
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]  (2 multiplications)
    # Total 4 multiplications, all essential. Rank = 4 per column.
    #
    # So rank = 28. Let me build this.

    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0.0] * rank for _ in range(m * p)]   # 14 x 28
    W = [[0.0] * rank for _ in range(p * n)]   # 14 x 28

    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
