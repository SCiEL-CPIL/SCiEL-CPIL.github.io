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
    
    # Kronecker products give 16x49 matrices
    U4_raw = np.kron(U2, U2)  # rows: (r_out*2+c_out)*4 + (r_in*2+c_in)
    V4_raw = np.kron(V2, V2)  # rows: (j_out*2+k_out)*4 + (j_in*2+k_in)
    W4_raw = np.kron(W2, W2)  # rows: (k_out*2+r_out)*4 + (k_in*2+r_in)
    
    # We need to permute rows to match the required indexing:
    # U: row index = i*4 + j where i=row of A, j=col of A
    #   In Kronecker: outer = i_out*2+j_out, inner = i_in*2+j_in
    #   So K = (i_out*2+j_out)*4 + (i_in*2+j_in)
    #   We need S = i*4 + j = (i_out*2+i_in)*4 + (j_out*2+j_in)
    #   So we need: perm_U[S] = K where S = i*4+j
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        K = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        perm_U[S] = K
    
    # V: row index = j*4 + k where j=row of B, k=col of B
    #   In Kronecker: outer = j_out*2+k_out, inner = j_in*2+k_in
    #   K = (j_out*2+k_out)*4 + (j_in*2+k_in)
    #   S = j*4 + k = (j_out*2+j_in)*4 + (k_out*2+k_in)
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = K
    
    # W: row index = k*4 + i where k=col of C, i=row of C
    #   In Kronecker: outer = k_out*2+i_out, inner = k_in*2+i_in
    #   K = (k_out*2+i_out)*4 + (k_in*2+i_in)
    #   S = k*4 + i = (k_out*2+k_in)*4 + (i_out*2+i_in)
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        K = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
        perm_W[S] = K
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Convert to list of lists as required
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
