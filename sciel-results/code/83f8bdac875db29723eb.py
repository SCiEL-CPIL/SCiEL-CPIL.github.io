"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    # Build left and right sides
    # LHS[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # RHS[k,n+i] = W[k*n+i, c] -- but we need per-c check
    # Actually the condition is: for each c, the tensor T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c] should equal W[k*n+i,c]
    # Wait, that's not right. The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # No, that's not standard. Let me re-read.
    
    # Standard matrix multiplication tensor: T[i,j,k] = 1 if multiplying A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a 3-way tensor decomposition (CP rank-r).
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And the entries should be such that this holds exactly.
    
    # Let's verify by checking the full tensor
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err < tol

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 3*5*5 = 75
    
    # For 3x5x5, the minimum rank is known to be at most 75 (schoolbook).
    # There may exist lower-rank decompositions, but finding them is hard.
    # Let's try to find a rank-72 or lower decomposition using known results.
    
    # Actually, for matrix multiplication, the tensor rank of M_{n,m,p} is at most n*m*p.
    # Known lower bounds: for M_{3,5,5}, the rank is at least max(n*m, m*p, n*p) = max(15, 25, 15) = 25.
    # But better bounds exist. The Strassen-like algorithms can reduce rank.
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # This gives rank 75, which is the seed.
    
    rank = rank_schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
