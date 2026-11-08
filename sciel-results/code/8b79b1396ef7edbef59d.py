"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    W uses trace convention: W[k*n+i, r]
    """
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook decomposition (rank = n*m*p = 40)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        # C[i,k] should equal sum_j A[i,j] * B[j,k]
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For random A, B, check that U^T * (A kron I) * ... matches
        # Simpler: check that the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] equals delta structure
        # Actually the tensor for matmul is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
        # Wait no. The matrix multiplication tensor T has T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
        # So T[i,j,k] = 1 for all valid i,j,k
        # And the decomposition is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Let's verify: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback: pure schoolbook
        pass
    
    # Convert to lists if needed
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
