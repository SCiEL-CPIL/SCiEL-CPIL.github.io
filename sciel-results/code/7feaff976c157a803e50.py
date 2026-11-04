import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 100
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def snap_to_half_integer(X):
        return [[round(x * 2) / 2 for x in row] for row in X]
    
    def solve_U(U, V, W, rank):
        U_new = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        return U_new
    
    def solve_V(U, V, W, rank):
        V_new = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        return V_new
    
    def solve_W(U, V, W, rank):
        W_new = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        return W_new
    
    def objective(U, V, W, rank):
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1.0) ** 2
        return residual
    
    def try_rank_als(rank, max_iters=100, n_restarts=5):
        best_residual = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(restart * 7919 + 12345)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            prev_res = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = solve_U(U, V, W, rank)
                V = solve_V(U, V, W, rank)
                W = solve_W(U, V, W, rank)
                
                res = objective(U, V, W, rank)
                if res < best_residual:
                    best_residual = res
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if abs(prev_res - res) < 1e-10:
                    break
                prev_res = res
            
            if best_residual < 1e-8:
                break
        
        if best_solution is not None and best_residual < 1e-8:
            U, V, W = best_solution
            U_snap = snap_to_half_integer(U.tolist())
            V_snap = snap_to_half_integer(V.tolist())
            W_snap = snap_to_half_integer(W.tolist())
            if verify(U_snap, V_snap, W_snap, rank):
                return U_snap, V_snap, W_snap
            # Check if raw values are close to half-integers
            U_check = np.array(U_snap)
            V_check = np.array(V_snap)
            W_check = np.array(W_snap)
            if (np.abs(U_check - U) < 0.01).all() and (np.abs(V_check - V) < 0.01).all() and (np.abs(W_check - W) < 0.01).all():
                return U_snap, V_snap, W_snap
            return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Try ranks from 24 up to 35
    for rank in range(24, 36):
        if time.time() > deadline - 15:
            break
        result = try_rank_als(rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
