import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 140
    
    rank = 92
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        U[i0*m+j0, col] = 1
        U[(i0+1)*m+(j0+1), col] = 1
        V[j0*p+k0, col] = 1
        V[(j0+1)*p+(k0+1), col] = 1
        W[k0*n+i0, col] = 1
        W[(k0+1)*n+(i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m+j0, col] = 1
        U[(i0+1)*m+(j0+1), col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+(i0+1), col] = 1
        col += 1
        
        U[i0*m+j0, col] = 1
        V[j0*p+(k0+1), col] = 1
        V[(j0+1)*p+(k0+1), col] = -1
        W[(k0+1)*n+i0, col] = 1
        col += 1
        
        U[(i0+1)*m+(j0+1), col] = 1
        V[(j0+1)*p+k0, col] = 1
        V[j0*p+k0, col] = -1
        W[k0*n+i0, col] = 1
        W[k0*n+(i0+1), col] = 1
        col += 1
        
        U[i0*m+j0, col] = 1
        U[i0*m+(j0+1), col] = 1
        V[(j0+1)*p+(k0+1), col] = 1
        W[k0*n+i0, col] = -1
        W[(k0+1)*n+i0, col] = 1
        col += 1
        
        U[(i0+1)*m+j0, col] = 1
        U[i0*m+j0, col] = -1
        V[j0*p+k0, col] = 1
        V[j0*p+(k0+1), col] = 1
        W[(k0+1)*n+(i0+1), col] = 1
        col += 1
        
        U[i0*m+(j0+1), col] = 1
        U[(i0+1)*m+(j0+1), col] = -1
        V[(j0+1)*p+k0, col] = 1
        V[(j0+1)*p+(k0+1), col] = 1
        W[k0*n+i0, col] = 1
        col += 1
    
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                if time.time() > deadline:
                    break
                add_strassen(i0, j0, k0)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if j == 4 or k == 4:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
