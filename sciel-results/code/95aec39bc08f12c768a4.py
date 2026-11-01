import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Precompute indices for speed
    # For each block l (0..7) and strassen step s (0..6)
    # u indices are 0..3
    # v indices: 2*l, 2*l+1, 16+2*l, 16+2*l+1
    # w indices: 4*l, 4*l+2, 4*l+1, 4*l+3
    
    for l in range(8):
        # Precompute v and w row indices for this block
        b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
        c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
        
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            # strassen_U[s] is a list of 4 values
            U[0, col] = strassen_U[s][0]
            U[1, col] = strassen_U[s][1]
            U[2, col] = strassen_U[s][2]
            U[3, col] = strassen_U[s][3]
            
            # V
            V[b_rows[0], col] = strassen_V[s][0]
            V[b_rows[1], col] = strassen_V[s][1]
            V[b_rows[2], col] = strassen_V[s][2]
            V[b_rows[3], col] = strassen_V[s][3]
            
            # W
            W[c_rows[0], col] = strassen_W[s][0]
            W[c_rows[1], col] = strassen_W[s][1]
            W[c_rows[2], col] = strassen_W[s][2]
            W[c_rows[3], col] = strassen_W[s][3]
                
    # Verification
    # We check T[i*m+j, j*p+k, k*n+i] == 1
    # Use vectorized dot product for each (i,j,k)
    is_valid = True
    for i in range(n):
        for j in range(m):
            u = i * m + j
            for k in range(p):
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
