import numpy as np

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # 1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        # 2
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        # 3
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        # 4
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        # 5
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        # 6
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        # 7
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1

    # Block 1: j=2 (A[:, 2] * B[2, :]) - Rank 10
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2a: j in {0,1}, k in {0,1} (M_{2,2,2}) - Rank 7
    add_strassen(0, 0, 0)
    
    # Block 2b: j in {0,1}, k in {2,3} (M_{2,2,2}) - Rank 7
    add_strassen(0, 0, 2)
    
    # Block 2c: j in {0,1}, k in {4} (M_{2,2,1}) - Rank 4
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
