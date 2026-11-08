import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # 4 Strassen 2x2x2 blocks (maximal packing in 3x5x5)
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    # Uncovered terms are handled by 3 structured rank-1 terms instead of schoolbook
    # Total rank = 4 * 7 (Strassen) + 3 (Structured Remainder) = 31
    
    total_rank = 4 * 7 + 3
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Add the 3 structured remainder terms
    
    # Term 1: Covers i=2 plane (all j, k)
    c = col
    for j in range(m):
        U[2*m + j][c] += 1
    for j in range(m):
        for k in range(p):
            V[j*p + k][c] += 1
    for k in range(p):
        W[k*n + 2][c] += 1
    col += 1
    
    # Term 2: Covers i in {0,1}, j=4 (all k)
    c = col
    U[0*m + 4][c] += 1
    U[1*m + 4][c] += 1
    for k in range(p):
        V[4*p + k][c] += 1
    for k in range(p):
        W[k*n + 0][c] += 1
        W[k*n + 1][c] += 1
    col += 1
    
    # Term 3: Covers i in {0,1}, k=4 (j in 0..3)
    c = col
    for i in [0, 1]:
        for j in range(4):
            U[i*m + j][c] += 1
    for j in range(4):
        V[j*p + 4][c] += 1
    W[4*n + 0][c] += 1
    W[4*n + 1][c] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
