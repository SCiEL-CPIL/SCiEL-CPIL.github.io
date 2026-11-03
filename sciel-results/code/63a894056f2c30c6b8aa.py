import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 24
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Try to find a rank-24 solution using local search
    # This is a heuristic and may not succeed
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        rank = target_rank
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to help convergence
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        
        # Local search
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Compute error
            # T[i*m+j, j*p+k, k*n+i] should be 1
            # Error = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1
            err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        val = np.dot(U[idx_u], V[idx_v]) * W[idx_w] # This is not correct
                        # Actually, the term is sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
                        # Let's compute the sum
                        prod = U[idx_u] * V[idx_v] * W[idx_w]
                        s = np.sum(prod)
                        err += (s - 1.0) ** 2
                        
                        # Gradient
                        grad_U[idx_u] += 2 * (s - 1.0) * V[idx_v] * W[idx_w]
                        grad_V[idx_v] += 2 * (s - 1.0) * U[idx_u] * W[idx_w]
                        grad_W[idx_w] += 2 * (s - 1.0) * U[idx_u] * V[idx_v]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if err < 1e-6:
                break
        
        if err < 1e-4:
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check if rounded solution is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        s = np.sum(U_r[idx_u] * V_r[idx_v] * W_r[idx_w])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                best_U = [[float(x) for x in row] for row in U_r]
                best_V = [[float(x) for x in row] for row in V_r]
                best_W = [[float(x) for x in row] for row in W_r]
                best_rank = rank
                break # Found a good solution
    
    return {"U": best_U, "V": best_V, "W": best_W}
