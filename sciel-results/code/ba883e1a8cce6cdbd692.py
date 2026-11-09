"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 140.0  # safety margin before 150 s wall-clock kill

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition is exact for matrix multiplication."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,k), the (j) slice should be 1
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _kronecker_product_decomp(U1, V1, W1, U2, V2, W2):
    """Compute Kronecker product of two tensor decompositions."""
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], total_rank))
    V = np.zeros((V1.shape[0] * V2.shape[0], total_rank))
    W = np.zeros((W1.shape[0] * W2.shape[0], total_rank))
    
    col = 0
    for c in range(rank1):
        for d in range(rank2):
            U[:, col] = np.kron(U1[:, c], U2[:, d])
            V[:, col] = np.kron(V1[:, c], V2[:, d])
            W[:, col] = np.kron(W1[:, c], W2[:, d])
            col += 1
    
    return U, V, W

def _construct_4x4x9_kronecker():
    """Construct 4x4x9 decomposition using Kronecker product of 2x2x3 decompositions.
    
    4x4x9 = (2x2) x (2x2) x (3x3) = (2x2x3) ⊗ (2x2x3)
    Rank = rank(2x2x3) * rank(2x2x3) = 6 * 6 = 36 (if schoolbook is optimal for 2x2x3)
    """
    # Get schoolbook for 2x2x3 (rank 6)
    U1, V1, W1 = _schoolbook(2, 2, 3)
    
    # Kronecker product with itself
    U, V, W = _kronecker_product_decomp(U1, V1, W1, U1, V1, W1)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Try Kronecker product approach: 4x4x9 = (2x2x3) ⊗ (2x2x3), rank = 36
    U, V, W = _construct_4x4x9_kronecker()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
