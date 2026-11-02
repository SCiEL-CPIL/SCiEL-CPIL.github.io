import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def get_residuals(U, V, W):
        res = {}
        for (i, j, k) in triplets:
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[(i, j, k)] = val - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return float(np.sum([r*r for r in res.values()]))

    def optimize(U, V, W, max_iters=300, tol=1e-10, deadline=None):
        lr = 0.02
        best_err = total_error(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            res = get_residuals(U, V, W)
            err = total_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k), r in res.items():
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                dU[ui] += 2 * r * V[vi] * W[wi]
                dV[vi] += 2 * r * U[ui] * W[wi]
                dW[wi] += 2 * r * U[ui] * V[vi]
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            lr *= 0.995
            
        return best_U, best_V, best_W

    start_time = time.time()
    deadline = start_time + 140
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for r in range(40, 49):
        if time.time() > deadline:
            break
            
        num_restarts = 3
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            U_init = np.zeros((n*m, r))
            V_init = np.zeros((m*p, r))
            W_init = np.zeros((p*n, r))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U_init[u_idx(i, j), col] = 1.0
                        V_init[v_idx(j, k), col] = 1.0
                        W_init[w_idx(k, i), col] = 1.0
                        col += 1
                    if col >= r:
                        break
                if col >= r:
                    break
            if col >= r:
                break
            
            noise_level = 0.03
            U_init += noise_level * np.random.randn(n*m, r)
            V_init += noise_level * np.random.randn(m*p, r)
            W_init += noise_level * np.random.randn(p*n, r)
            
            U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, max_iters=150, tol=1e-10, deadline=deadline)
            
            err = total_error(U_opt, V_opt, W_opt)
            
            if err < best_err:
                best_err = err
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                best_rank = r
            
            if err < 1e-8:
                return {
                    "U": U_opt.tolist(),
                    "V": V_opt.tolist(),
                    "W": W_opt.tolist()
                }
    
    if best_U is not None and best_err < 1e-4:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
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
