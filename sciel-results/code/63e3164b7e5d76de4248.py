import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank estimate: 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1) = 65
    rank = 65
    
    # Initialize matrices
    # U: (n*m, rank) -> (16, 65)
    # V: (m*p, rank) -> (20, 65)
    # W: (p*n, rank) -> (20, 65)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Basis for 2x2x2
    # u, v, w are length 4 vectors representing 2x2 matrices
    # u[i*2+j] corresponds to A[i,j]
    # v[i*2+j] corresponds to B[i,j]
    # w[i*2+j] corresponds to C[i,j]
    
    strassen_basis = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Helper to map local 2x2 block index to global indices
    # local_i, local_j in {0,1}
    # Global i, j in {0..3}
    # Block (0,0): i in 0,1; j in 0,1
    # Block (0,1): i in 0,1; j in 2,3
    # Block (1,0): i in 2,3; j in 0,1
    # Block (1,1): i in 2,3; j in 2,3
    
    def get_global_indices(local_i, local_j, dim=4):
        # dim is 4, block size 2
        # local_i is row block index, local_j is col block index
        start_i = local_i * 2
        start_j = local_j * 2
        i_range = range(start_i, start_i + 2)
        j_range = range(start_j, start_j + 2)
        return i_range, j_range

    # Add Strassen terms
    for u_local, v_local, w_local in strassen_basis:
        # u_local is for A (4x4)
        # v_local is for B (4x4 part, cols 0-3)
        # w_local is for C (4x4 part, cols 0-3)
        
        # Fill U
        for li in range(2):
            for lj in range(2):
                val = u_local[li * 2 + lj]
                if val != 0:
                    i_range, j_range = get_global_indices(li, lj)
                    for i in i_range:
                        for j in j_range:
                            U[i * m + j][col] = val
        
        # Fill V (for B, cols 0-3)
        # B is 4x5. We are using the first 4 columns (0-3).
        # V index is j * p + k.
        # Here j is row of B (0-3), k is col of B (0-3).
        for li in range(2):
            for lj in range(2):
                val = v_local[li * 2 + lj]
                if val != 0:
                    i_range, j_range = get_global_indices(li, lj)
                    # i_range corresponds to rows of B (j in global)
                    # j_range corresponds to cols of B (k in global)
                    for j in i_range:
                        for k in j_range:
                            V[j * p + k][col] = val
        
        # Fill W (for C, cols 0-3)
        # C is 4x5. We are using the first 4 columns (0-3).
        # W index is k * n + i.
        # Here i is row of C (0-3), k is col of C (0-3).
        for li in range(2):
            for lj in range(2):
                val = w_local[li * 2 + lj]
                if val != 0:
                    i_range, j_range = get_global_indices(li, lj)
                    # i_range corresponds to rows of C (i in global)
                    # j_range corresponds to cols of C (k in global)
                    for i in i_range:
                        for k in j_range:
                            W[k * n + i][col] = val
        
        col += 1
    
    # Add Schoolbook terms for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 16
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

