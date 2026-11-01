"""Rewritten program for matrix multiplication 2x4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook solution
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W
    
    # Try various structured constructions
    constructions = []
    
    # Construction 1: rank = p*n
    r = p * n
    if r < best_rank:
        U_c = np.zeros((n*m, r))
        V_c = np.zeros((m*p, r))
        W_c = np.zeros((p*n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_c[i*m+j, k*n+i] = 1.0
                    V_c[j*p+k, k*n+i] = 1.0
                    W_c[k*n+i, k*n+i] = 1.0
        if _check(U_c, V_c, W_c, n, m, p):
            best_rank = r
            best_U, best_V, best_W = U_c.tolist(), V_c.tolist(), W_c.tolist()
    
    # Construction 2: rank = m*p
    r = m * p
    if r < best_rank:
        U_c = np.zeros((n*m, r))
        V_c = np.zeros((m*p, r))
        W_c = np.zeros((p*n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_c[i*m+j, j*p+k] = 1.0
                    V_c[j*p+k, j*p+k] = 1.0
                    W_c[k*n+i, j*p+k] = 1.0
        if _check(U_c, V_c, W_c, n, m, p):
            best_rank = r
            best_U, best_V, best_W = U_c.tolist(), V_c.tolist(), W_c.tolist()
    
    # Construction 3: rank = n*m
    r = n * m
    if r < best_rank:
        U_c = np.zeros((n*m, r))
        V_c = np.zeros((m*p, r))
        W_c = np.zeros((p*n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_c[i*m+j, i*m+j] = 1.0
                    V_c[j*p+k, i*m+j] = 1.0
                    W_c[k*n+i, i*m+j] = 1.0
        if _check(U_c, V_c, W_c, n, m, p):
            best_rank = r
            best_U, best_V, best_W = U_c.tolist(), V_c.tolist(), W_c.tolist()
    
    # Try to find even lower rank via search
    if time.time() < deadline and best_rank > 1:
        # Try ranks from 1 to best_rank-1 with random search
        for rank in range(1, best_rank):
            if time.time() > deadline:
                break
            found = _random_search(rank, n, m, p, deadline)
            if found:
                U_f, V_f, W_f = found
                best_rank = rank
                best_U, best_V, best_W = U_f.tolist(), V_f.tolist(), W_f.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _check(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _random_search(rank, n, m, p, deadline):
    """Try to find a rank-r decomposition via random search."""
    for attempt in range(200):
        if time.time() > deadline:
            return None
        np.random.seed(attempt * 12345 + rank)
        U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
        if _check(U, V, W, n, m, p):
            return U, V, W
    return None
# EVOLVE-BLOCK-END
