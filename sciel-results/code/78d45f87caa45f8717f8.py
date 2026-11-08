"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if (i,j,k) is valid, 0 otherwise
    # For matrix multiplication, the tensor is T[i*m+j, j*p+k, k*n+i] = 1
    # Actually, the decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (since all combinations are valid in matmul)
    
    # Wait, in matrix multiplication, every (i,j,k) triple contributes, so the tensor is all-ones.
    # Let me verify:
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_s = [[0] * rank_schoolbook for _ in range(n * m)]
        V_s = [[0] * rank_schoolbook for _ in range(m * p)]
        W_s = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
