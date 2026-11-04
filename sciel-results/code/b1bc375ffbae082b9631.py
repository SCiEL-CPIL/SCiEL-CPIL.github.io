import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Build target tensor T of shape (N_U, N_V, N_W)
    T = np.zeros((N_U, N_V, N_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # First, try a very fast, targeted search for Rank 23
    # We assume the decomposition might be close to schoolbook (24)
    # We try to merge two terms.
    # This is a heuristic.
    
    best_rank = 24
    best_decomp = None
    
    # Try Rank 23
    if time.time() < deadline - 5.0:
        R = 23
        # Random restarts
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with schoolbook terms, then perturb
            U = np.zeros((N_U, R))
            V = np.zeros((N_V, R))
            W = np.zeros((N_W, R))
            
            # Fill first 22 columns with schoolbook-like structure
            # This is complex to do correctly.
            # Instead, use a simple random search with integer constraints.
            U = np.random.choice([-1, 0, 1], size=(N_U, R))
            V = np.random.choice([-1, 0, 1], size=(N_V, R))
            W = np.random.choice([-1, 0, 1], size=(N_W, R))
            
            # Check if U * V * W == T
            # This is O(N_U * N_V * N_W * R)
            # 24 * 12 * 8 * 23 ~ 50k ops. Very fast.
            
            error = 0.0
            for a in range(N_U):
                for b in range(N_V):
                    for c in range(N_W):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        target = T[a, b, c]
                        if abs(val - target) > 1e-6:
                            error += abs(val - target)
                            if error > 1e-6:
                                break
                    if error > 1e-6:
                        break
                if error > 1e-6:
                    break
                    
            if error < 1e-6:
                best_rank = R
                best_decomp = (U, V, W)
                break

    if best_decomp is not None:
        U, V, W = best_decomp
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
