import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from low to high
    for rank in range(20, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a rank-r decomposition
        # U: (n*m) x r = 6 x r
        # V: (m*p) x r = 24 x r
        # W: (p*n) x r = 16 x r
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Use random initialization and Newton's method
        for restart in range(30):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 1000 + rank)
            U = np.random.uniform(-2, 2, (n*m, rank))
            V = np.random.uniform(-2, 2, (m*p, rank))
            W = np.random.uniform(-2, 2, (p*n, rank))
            
            # Newton's method to solve for exact solution
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current T
                T = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U[:, c]  # 6
                    v_c = V[:, c]  # 24
                    w_c = W[:, c]  # 16
                    for i in range(n):
                        for j in range(m):
                            u_val = u_c[i*m + j]
                            if u_val == 0:
                                continue
                            for k in range(p):
                                v_val = v_c[j*p + k]
                                if v_val == 0:
                                    continue
                                w_val = w_c[k*n + i]
                                T[i, j, k] += u_val * v_val * w_val
                
                # Residual
                R = T - 1.0
                err = np.max(np.abs(R))
                if err < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = np.zeros((n, m, p))
                    for c in range(rank):
                        u_c = U_r[:, c]
                        v_c = V_r[:, c]
                        w_c = W_r[:, c]
                        for i in range(n):
                            for j in range(m):
                                u_val = u_c[i*m + j]
                                if u_val == 0:
                                    continue
                                for k in range(p):
                                    v_val = v_c[j*p + k]
                                    if v_val == 0:
                                        continue
                                    w_val = w_c[k*n + i]
                                    T_check[i, j, k] += u_val * v_val * w_val
                    
                    if np.max(np.abs(T_check - 1.0)) < 1e-9:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                    
                    # If rounding didn't work, try to adjust
                    # For now, break and try next restart
                    break
                
                # Compute Jacobian (approximate)
                # This is expensive, so use finite differences or skip
                # Instead, use a simple gradient step
                lr = 0.01
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            u_idx = i*m + j
                            grad_u = 0
                            for k in range(p):
                                v_val = V[j*p + k, c]
                                w_val = W[k*n + i, c]
                                grad_u += v_val * w_val * R[i, j, k]
                            U[u_idx, c] -= lr * grad_u
                            
                    for j in range(m):
                        for k in range(p):
                            v_idx = j*p + k
                            grad_v = 0
                            for i in range(n):
                                u_val = U[i*m + j, c]
                                w_val = W[k*n + i, c]
                                grad_v += u_val * w_val * R[i, j, k]
                            V[v_idx, c] -= lr * grad_v
                            
                    for k in range(p):
                        for i in range(n):
                            w_idx = k*n + i
                            grad_w = 0
                            for j in range(m):
                                u_val = U[i*m + j, c]
                                v_val = V[j*p + k, c]
                                grad_w += u_val * v_val * R[i, j, k]
                            W[w_idx, c] -= lr * grad_w
                        
            # After Newton iterations, check if we got close
            T = np.zeros((n, m, p))
            for c in range(rank):
                u_c = U[:, c]
                v_c = V[:, c]
                w_c = W[:, c]
                for i in range(n):
                    for j in range(m):
                        u_val = u_c[i*m + j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = v_c[j*p + k]
                            if v_val == 0:
                                continue
                            w_val = w_c[k*n + i]
                            T[i, j, k] += u_val * v_val * w_val
            
            err = np.max(np.abs(T - 1.0))
            if err < 1e-8:
                # Try rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_check = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U_r[:, c]
                    v_c = V_r[:, c]
                    w_c = W_r[:, c]
                    for i in range(n):
                        for j in range(m):
                            u_val = u_c[i*m + j]
                            if u_val == 0:
                                continue
                            for k in range(p):
                                v_val = v_c[j*p + k]
                                if v_val == 0:
                                    continue
                                w_val = w_c[k*n + i]
                                T_check[i, j, k] += u_val * v_val * w_val
                
                if np.max(np.abs(T_check - 1.0)) < 1e-9:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
    
    # Fallback to schoolbook
    rank = rank_schoolbook
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
