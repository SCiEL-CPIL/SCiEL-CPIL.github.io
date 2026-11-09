"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Attempt rank-20 decomposition using structured basis vectors
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 20 rank-1 components with integer coefficients
    # Each component r defines vectors u_r, v_r, w_r
    # We need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
    
    # Construct using a systematic approach
    # For each of the 20 terms, assign patterns to U, V, W
    
    # Term 0-5: cover i=0 cases
    for r in range(6):
        # U: i=0, j=r%3
        U[0 * m + (r % 3)][r] = 1
        # V: j=r%3, k=r%10
        V[(r % 3) * p + (r % 10)][r] = 1
        # W: k=r%10, i=0
        W[(r % 10) * n + 0][r] = 1
    
    # Term 6-11: cover i=1 cases  
    for r in range(6, 12):
        rr = r - 6
        # U: i=1, j=rr%3
        U[1 * m + (rr % 3)][r] = 1
        # V: j=rr%3, k=rr%10
        V[(rr % 3) * p + (rr % 10)][r] = 1
        # W: k=rr%10, i=1
        W[(rr % 10) * n + 1][r] = 1
    
    # Term 12-19: additional terms to complete the decomposition
    # These need to handle the remaining constraints
    for r in range(12, 20):
        rr = r - 12
        # Distribute remaining coverage
        i_idx = rr % 2
        j_idx = (rr // 2) % 3
        k_idx = (rr // 6) % 10
        U[i_idx * m + j_idx][r] = 1
        V[j_idx * p + k_idx][r] = 1
        W[k_idx * n + i_idx][r] = 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fall back to trivial decomposition if verification fails
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
