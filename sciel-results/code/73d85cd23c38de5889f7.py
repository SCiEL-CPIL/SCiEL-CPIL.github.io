"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds safety margin before 150s kill

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using the
    Kronecker product of Strassen's rank-7 decomposition for 2x2.
    Direct element-wise construction avoids np.kron + permutation overhead.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank-64
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                c = i * m + j
                U[i*m + j][c] = 1
                for k in range(p):
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (input A, rows indexed by i*2+j)
    # V2: 4x7 (input B, rows indexed by j*2+k)
    # W2: 4x7 (output C, rows indexed by k*2+i)
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1, 0,  1,  0,  1]
    ], dtype=int)
    
    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=int)
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Precompute the 7x7 outer product structure
    # For each pair (alpha, beta) in 0..6, column = alpha*7 + beta
    
    # Build U: row = i*4+j, decompose i=(i1,i0), j=(j1,j0)
    # U[i*4+j, alpha*7+beta] = U2[i1*2+j1, alpha] * U2[i0*2+j0, beta]
    for i in range(4):
        i1, i0 = divmod(i, 2)
        for j in range(4):
            j1, j0 = divmod(j, 2)
            row_u = i * 4 + j
            u_block = U2[i1 * 2 + j1]  # shape (7,)
            u_entry = U2[i0 * 2 + j0]  # shape (7,)
            # Outer product: (7,7) -> flatten to 49
            U[row_u] = np.outer(u_block, u_entry).flatten()
    
    # Build V: row = j*4+k, decompose j=(j1,j0), k=(k1,k0)
    # V[j*4+k, alpha*7+beta] = V2[j1*2+k1, alpha] * V2[j0*2+k0, beta]
    for j in range(4):
        j1, j0 = divmod(j, 2)
        for k in range(4):
            k1, k0 = divmod(k, 2)
            row_v = j * 4 + k
            v_block = V2[j1 * 2 + k1]
            v_entry = V2[j0 * 2 + k0]
            V[row_v] = np.outer(v_block, v_entry).flatten()
    
    # Build W: row = k*4+i, decompose k=(k1,k0), i=(i1,i0)
    # W[k*4+i, alpha*7+beta] = W2[k1*2+i1, alpha] * W2[k0*2+i0, beta]
    for k in range(4):
        k1, k0 = divmod(k, 2)
        for i in range(4):
            i1, i0 = divmod(i, 2)
            row_w = k * 4 + i
            w_block = W2[k1 * 2 + i1]
            w_entry = W2[k0 * 2 + i0]
            W[row_w] = np.outer(w_block, w_entry).flatten()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
