"""Matrix multiplication 2x2 * 2x7 with improved tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    # We are multiplying A (2x2) by B (2x7) to get C (2x7).
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,...,6}
    #
    # Naive rank = 2*2*7 = 28.
    #
    # We can exploit the fact that for each fixed i, the row C[i,:] is a linear
    # combination of the rows of B with coefficients A[i,0] and A[i,1].
    #
    # A better approach: use the structure that B has 2 rows and 7 columns.
    # For each column k, we need to compute C[0,k] and C[1,k].
    #
    # Let's use a decomposition based on grouping by the column index.
    # For each k in {0,...,6}, we need:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # That's 2 bilinear forms per column, each with 2 terms = 4 terms per column.
    # Total = 7 * 4 = 28 (same as naive).
    #
    # Can we do better? Let's try to share some computations.
    #
    # Idea: For each pair of columns (k1, k2), we can compute:
    #   S0 = B[0,k1] + B[0,k2]
    #   S1 = B[1,k1] + B[1,k2]
    # Then:
    #   C[0,k1] + C[0,k2] = A[0,0]*S0 + A[0,1]*S1
    #   C[1,k1] + C[1,k2] = A[1,0]*S0 + A[1,1]*S1
    #
    # But we still need the individual C values, so this doesn't directly help
    # unless we can recover them from sums.
    #
    # Alternative approach: Use the fact that we only have 4 entries in A.
    # Let's define intermediate quantities:
    #   P00 = A[0,0], P01 = A[0,1], P10 = A[1,0], P11 = A[1,1]
    #
    # For each column k:
    #   C[0,k] = P00*B[0,k] + P01*B[1,k]
    #   C[1,k] = P10*B[0,k] + P11*B[1,k]
    #
    # Total bilinear terms = 7 * 4 = 28.
    #
    # Let's try a different strategy: express the multiplication as a combination
    # of fewer bilinear forms by using linearity in B.
    #
    # Actually, let's just try to find a rank-14 decomposition by pairing columns.
    # For each k, define:
    #   V_k = (B[0,k], B[1,k])  -- a 2-vector
    # Then C[:,k] = A @ V_k
    #
    # The tensor is a sum over k of (identity on A) ⊗ (e_k on B) ⊗ (e_k on C)
    # where the "identity on A" part is actually the 2x2 matrix multiplication.
    #
    # Let me just provide the naive decomposition. Finding a better one requires
    # more sophisticated analysis that I can't do reliably here.
    
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
