"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 15s margin before 160s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p
    # Known lower bounds: for 4x4x5, we can try various ranks
    
    # Use a randomized approach: try to find a decomposition of decreasing rank
    # by solving linear systems
    
    def try_rank(r, max_iters=50, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random matrices
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want T[i,j,k] = 1 if this corresponds to a valid multiplication term
        
        # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
        
        # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            # Gradient descent step
            lr = 0.01
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        grad = 2.0 * (val - 1.0)
                        for c in range(r):
                            dU[i*m+j, c] += grad * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += grad * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += grad * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < 1e-8:
            # Snap to nearest integers/half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            # Verify
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_snap[i*m+j, c] * V_snap[j*p+k, c] * W_snap[k*n+i, c] for c in range(r))
                        if abs(val - 1.0) > 1e-6:
                            verified = False
                            break
                    if not verified:
                        break
                if not verified:
                    break
            
            if verified:
                return U_snap, V_snap, W_snap, best_err
        
        return None
    
    # Try ranks from low to high
    for r in range(12, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=30, seed=r)
        if result is not None:
            U_snap, V_snap, W_snap, err = result
            best = {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist(),
                "rank": r
            }
            break
    
    # If no better found, return schoolbook
    if best["rank"] == rank_schoolbook:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    else:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
