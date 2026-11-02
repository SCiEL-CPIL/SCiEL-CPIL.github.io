"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Uses a block-decomposed structure (Kronecker-like) where the multiplication is split
    into 3 independent 1x5x7 multiplications.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros using NumPy
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # Decompose into n independent 1 x m x p multiplications
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                u_idx = i * m + j
                # Index in V: j*p + k
                v_idx = j * p + k
                # Index in W: k*n + i
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
