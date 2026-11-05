import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((nm, rank_schoolbook))
    V = np.zeros((mp, rank_schoolbook))
    W = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U = U.tolist()
    best_V = V.tolist()
    best_W = W.tolist()
    best_rank = rank_schoolbook
    
    start_time = time.time()
    time_limit = 120
    
    # Try to find a lower rank
    # We use a simple randomized approach
    for R in range(60, 72):
        if time.time() - start_time > time_limit - 10:
            break
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            U_rand = np.random.randn(nm, R)
            V_rand = np.random.randn(mp, R)
            W_rand = np.random.randn(pn, R)
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.sum(U_rand[:, idx_u] * V_rand[:, idx_v] * W_rand[:, idx_w])
                        if not np.isclose(val, 1.0, atol=1e-3):
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                # Check non-linked
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            for j2 in range(m):
                                for k2 in range(p):
                                    idx_v = j2 * p + k2
                                    for i2 in range(n):
                                        idx_w = k2 * n + i2
                                        if i == i2 and j == j2 and k == k2:
                                            continue
                                        val = np.sum(U_rand[:, idx_u] * V_rand[:, idx_v] * W_rand[:, idx_w])
                                        if not np.isclose(val, 0.0, atol=1e-3):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_rand.tolist()
                    best_rank = R
                    break
            if best_rank < rank_schoolbook:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
