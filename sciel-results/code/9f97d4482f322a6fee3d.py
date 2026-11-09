"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Indices in U: i*m + j
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    # Standard Strassen 2x2x2 tensors
    # u, v, w are length 4 vectors corresponding to (0,0), (0,1), (1,0), (1,1)
    # for u (i,j), v (j,k), w (k,i)
    
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Map local indices (0,1) to global indices
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    i_map = [0, 1]
    j_map = [0, 1]
    k_map = [0, 1]
    
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                val = s_u[r][li * 2 + lj]
                if val != 0:
                    gi = i_map[li]
                    gj = j_map[lj]
                    U[gi * m + gj, col] += val
        for lj in range(2):
            for lk in range(2):
                val = s_v[r][lj * 2 + lk]
                if val != 0:
                    gj = j_map[lj]
                    gk = k_map[lk]
                    V[gj * p + gk, col] += val
        for lk in range(2):
            for li in range(2):
                val = s_w[r][lk * 2 + li]
                if val != 0:
                    gk = k_map[lk]
                    gi = i_map[li]
                    W[gk * n + gi, col] += val
        col += 1
        
    # Strassen block 2: i in {0,1}, j in {0,1}, k in {2,3}
    i_map = [0, 1]
    j_map = [0, 1]
    k_map = [2, 3]
    
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                val = s_u[r][li * 2 + lj]
                if val != 0:
                    gi = i_map[li]
                    gj = j_map[lj]
                    U[gi * m + gj, col] += val
        for lj in range(2):
            for lk in range(2):
                val = s_v[r][lj * 2 + lk]
                if val != 0:
                    gj = j_map[lj]
                    gk = k_map[lk]
                    V[gj * p + gk, col] += val
        for lk in range(2):
            for li in range(2):
                val = s_w[r][lk * 2 + li]
                if val != 0:
                    gk = k_map[lk]
                    gi = i_map[li]
                    W[gk * n + gi, col] += val
        col += 1
        
    # Schoolbook block: i in {0,1}, j=2, k in {0,1,2,3}
    for i in range(2):
        for j in [2]:
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to list of lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
