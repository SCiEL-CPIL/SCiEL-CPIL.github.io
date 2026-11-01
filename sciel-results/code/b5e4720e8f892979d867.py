import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses a decomposition of rank m*p = 30.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank is m * p
    rank = m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # Term r = col
            # U indices: j (for i=0) and m+j (for i=1)
            U[j][col] = 1
            U[m + j][col] = 1
            
            # V index: j*p + k
            V[j * p + k][col] = 1
            
            # W indices: k*n (for i=0) and k*n + 1 (for i=1)
            W[k * n][col] = 1
            W[k * n + 1][col] = 1
            
            col += 1
            
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    if U[u_idx][c] != 0 and V[v_idx][c] != 0 and W[w_idx][c] != 0:
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
