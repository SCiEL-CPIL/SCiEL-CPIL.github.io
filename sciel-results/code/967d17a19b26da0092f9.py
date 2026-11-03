"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def verify_decomp(U, V, W, r):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += W[k*n+i, r_idx] * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = schoolbook()
    
    # Numerical search: random init + gradient-free optimization
    def search_rank(r, max_iters=5000):
        """Try to find decomposition of rank r."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Compute error
                err = 0.0
                grad_U = np.zeros((n*m, r))
                grad_V = np.zeros((m*p, r))
                grad_W = np.zeros((p*n, r))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += W[k*n+i, r_idx] * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                            e = val - 1.0
                            err += e * e
                            for r_idx in range(r):
                                grad_U[i*m+j, r_idx] += e * W[k*n+i, r_idx] * V[j*p+k, r_idx]
                                grad_V[j*p+k, r_idx] += e * W[k*n+i, r_idx] * U[i*m+j, r_idx]
                                grad_W[k*n+i, r_idx] += e * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                # Gradient step
                lr = 0.01 / (1 + it * 0.001)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if best_err < 1e-10:
                # Try to snap to half-integers
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                if verify_decomp(U_snap, V_snap, W_snap, r):
                    return U_snap, V_snap, W_snap, r
                if verify_decomp(best_U, best_V, best_W, r):
                    return best_U, best_V, best_W, r
                # Try more precise optimization from here
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                for it2 in range(2000):
                    if time.time() > deadline - 5:
                        break
                    grad_U = np.zeros((n*m, r))
                    grad_V = np.zeros((m*p, r))
                    grad_W = np.zeros((p*n, r))
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r_idx in range(r):
                                    val += W[k*n+i, r_idx] * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                                e = val - 1.0
                                err += e * e
                                for r_idx in range(r):
                                    grad_U[i*m+j, r_idx] += e * W[k*n+i, r_idx] * V[j*p+k, r_idx]
                                    grad_V[j*p+k, r_idx] += e * W[k*n+i, r_idx] * U[i*m+j, r_idx]
                                    grad_W[k*n+i, r_idx] += e * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    if err < 1e-14:
                        break
                    lr = 0.001 / (1 + it2 * 0.001)
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if verify_decomp(U_snap, V_snap, W_snap, r):
                        return U_snap, V_snap, W_snap, r
                if verify_decomp(U, V, W, r):
                    return U, V, W, r
        
        return None
    
    # Try ranks in increasing order
    for r in range(20, 63):
        if time.time() > deadline - 15:
            break
        result = search_rank(r)
        if result is not None:
            U, V, W, rank = result
            if rank < rank_best:
                rank_best = rank
                U_best, V_best, W_best = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(U_best[i, j]) for j in range(rank_best)] for i in range(n*m)]
    V_list = [[float(V_best[i, j]) for j in range(rank_best)] for i in range(m*p)]
    W_list = [[float(W_best[i, j]) for j in range(rank_best)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

