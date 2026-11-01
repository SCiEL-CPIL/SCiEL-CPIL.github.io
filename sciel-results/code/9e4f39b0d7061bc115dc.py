import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook rank 12
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try to find lower rank
    for r in [8, 9, 10, 11]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition_fast(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            if r == 8:
                break

    return best_result

def _find_decomposition_fast(n, m, p, r, deadline):
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(2000):
            if time.time() > deadline:
                return None
            
            # Compute vals
            # vals[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            
            loss = np.sum((vals - 1.0)**2)
            if loss < 1e-6:
                # Round
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                vals_r = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            vals_r[i, j, k] = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                
                if np.sum((vals_r - 1.0)**2) < 1e-5:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Gradient Descent
            # Gradient is small, use simple update
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = vals[i, j, k] - 1.0
                        grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            U -= 0.01 * grad_U
            V -= 0.01 * grad_V
            W -= 0.01 * grad_W
    return None
