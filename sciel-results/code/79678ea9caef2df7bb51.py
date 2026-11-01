import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    # Local indices: ii in {0,1}, jj in {0,1}, kk in {0,1}
    # Global indices: i = i_offset + ii, j = j_offset + jj, k = k_offset + kk
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0], -C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1
    
    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1], C[1,1]
    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1
    
    # M5: (A[0,0]+A[0,1])B[1,1] -> -C[0,0], C[0,1]
    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]
    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    start_time = time.time()
    
    # Rank 28 decomposition
    # Block 1: j=0, k=0..4 (2x1x5) -> Rank 10
    # Block 2: j=1,2, k=0,1 (2x2x2) -> Rank 7
    # Block 3: j=1,2, k=2 (2x2x1) -> Rank 4
    # Block 4: j=1,2, k=3,4 (2x2x2) -> Rank 7
    # Total: 10 + 7 + 4 + 7 = 28
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, k=0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
            
    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Deadline guard (though this is static, it satisfies the directive)
    if time.time() - start_time > 110:
         pass # Would return best found if search was dynamic, here it's static so we just return.

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
