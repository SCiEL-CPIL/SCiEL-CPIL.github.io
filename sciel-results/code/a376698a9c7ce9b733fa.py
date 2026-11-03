"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    
    # We need to decompose the matrix multiplication tensor.
    # The tensor T has indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The standard decomposition has rank n*m*p.
    # However, we can potentially reduce the rank by using bilinear methods.
    # For small matrices, we can search for a lower rank decomposition.
    
    # Let's try to find a decomposition with lower rank using a simple approach.
    # For 2x4 * 4x6, the schoolbook rank is 2*4*6 = 48.
    # We can try to use a decomposition that exploits the structure.
    
    # One approach: use the fact that matrix multiplication can be done with
    # fewer multiplications using algorithms like Strassen's, but for these
    # specific dimensions, we need to check if a lower rank exists.
    
    # For now, let's use the schoolbook decomposition as a baseline.
    # The rank is n*m*p = 48.
    
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Populate the decomposition factors
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i (Trace convention)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1.0
                V[v_idx][idx] = 1.0
                W[w_idx][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
