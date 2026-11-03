"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen decomposition for 2x2x2, Kronecker product for 4x4x4,
    and block assembly for 4x4x8.
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker product to get 4x4x4 decomposition
    # Rank 7 * 7 = 49
    U_444 = np.kron(U_222, U_222)
    V_444 = np.kron(V_222, V_222)
    W_444 = np.kron(W_222, W_222)
    
    # Assemble for 4x4x8
    # U is 16x98
    # V is 32x98
    # W is 32x98
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Block 1 (cols 0-3 of B and C)
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Block 2 (cols 4-7 of B and C)
    U[:, 49:98] = U_444
    V[16:32, 49:98] = V_444
    W[16:32, 49:98] = W_444
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
