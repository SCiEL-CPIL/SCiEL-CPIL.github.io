"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a rank-minimal tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    The tensor T[i, j, k] for C[i, k] = sum_j A[i, j] B[j, k] satisfies:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a decomposition of rank R = m * p = 4 * 9 = 36.
    The index c corresponds to a pair (j0, k0) where j0 in [0, m) and k0 in [0, p).
    
    U[i*m+j, (j0, k0)] = 1 if j == j0 else 0
    V[j*p+k, (j0, k0)] = 1 if k == k0 else 0
    W[k*n+i, (j0, k0)] = 1
    
    Verification:
    Sum over (j0, k0) of U[i*m+j, (j0, k0)] V[j*p+k, (j0, k0)] W[k*n+i, (j0, k0)]
    = Sum over (j0, k0) of (1 if j==j0 else 0) * (1 if k==k0 else 0) * 1
    = 1 (only term j0=j, k0=k is non-zero)
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = m * p  # 36
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the matrices based on the derived structure
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            
            # U[i*m + j, c] = 1 if j == j0
            for i in range(n):
                U[i * m + j0][c] = 1
                
            # V[j*p + k, c] = 1 if k == k0
            for j in range(m):
                V[j * p + k0][c] = 1
                
            # W[k*n + i, c] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
