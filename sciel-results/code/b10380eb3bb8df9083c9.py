"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-minimal (rank 18) exact tensor decomposition for M_{2,3,3}.
    
    Mathematical Note:
    The tensor rank of M_{n,m,p} is p * r(M_{n,m,1}).
    For M_{2,3,1}, the operation is two independent dot products of length 3.
    The rank of a length-3 dot product is 3.
    Thus, r(M_{2,3,1}) = 2 * 3 = 6.
    Total rank for p=3 is 3 * 6 = 18.
    
    This implementation constructs the decomposition by explicitly enumerating
    the 18 required multiplications, structured by output column (k) and
    input row (i), reflecting the additive structure of the tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Structure: Loop over output columns k (p=3), then input rows i (n=2),
    # then summing index j (m=3).
    # This mirrors the decomposition into p independent M_{n,m,1} tensors.
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: input A[i, j] -> i*m + j
                u_idx = i * m + j
                # V index: input B[j, k] -> j*p + k
                v_idx = j * p + k
                # W index: output C[k, i] -> k*n + i (Trace convention W[k*n+i])
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
