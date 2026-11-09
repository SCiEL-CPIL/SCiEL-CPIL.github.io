"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _construct_sub_decomposition(n, m, p, target_rank):
    """
    Finds a rank decomposition for n x m x p tensor (all ones)
    using ALS-like iteration.
    """
    # T[i, j, k] = 1
    # U shape (n*m, R)
    # V shape (m*p, R)
    # W shape (p*n, R)
    
    # We will use a simple gradient descent or ALS.
    # Given the small size, we can try random restarts.
    
    best_rank = None
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    for r in range(target_rank, target_rank + 5):
        for restart in range(10):
            rng = np.random.default_rng(restart)
            U = rng.normal(size=(n*m, r))
            V = rng.normal(size=(m*p, r))
            W = rng.normal(size=(p*n, r))
            
            # Normalize to prevent divergence
            for i in range(r):
                nU = np.linalg.norm(U[:, i])
                nV = np.linalg.norm(V[:, i])
                nW = np.linalg.norm(W[:, i])
                if nU > 1e-10: U[:, i] /= nU
                if nV > 1e-10: V[:, i] /= nV
                if nW > 1e-10: W[:, i] /= nW
            
            for it in range(200):
                # Update U
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # For fixed j, k:
                # T[i,j,k] = sum_r U[i,j,r] (V[j,k,r] W[k,i,r])
                # This is tricky because W depends on i.
                
                # Let's use a simple error calculation and gradient.
                # C = np.zeros((n, m, p))
                # for i in range(n):
                #   for j in range(m):
                #     for k in range(p):
                #       C[i,j,k] = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(r))
                
                # This is too slow in Python loops.
                # Vectorize:
                # U_exp: (n, m, R)
                # V_exp: (m, p, R)
                # W_exp: (p, n, R)
                
                U_exp = U.reshape(n, m, r)
                V_exp = V.reshape(m, p, r)
                W_exp = W.reshape(p, n, r)
                
                # C[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # We can compute this using einsum
                C = np.einsum('ijr,jkr,kir->ijk', U_exp, V_exp, W_exp)
                
                T = np.ones((n, m, p))
                err = np.linalg.norm(C - T)
                
                if err < 1e-6:
                    return r, U, V, W
                
                # Gradient descent step
                # dE/dU[i,j,r] = 2 * (C[i,j,k] - T[i,j,k]) * V[j,k,r] * W[k,i,r]
                # Sum over k.
                
                # dU[i,j,r] = sum_k (C[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
                # dU = np.einsum('ijk,jkr,kir->ijr', C - T, V_exp, W_exp)
                # dV = np.einsum('ijk,ijr,kir->jkr', C - T, U_exp, W_exp)
                # dW = np.einsum('ijk,ijr,jkr->kir', C - T, U_exp, V_exp)
                
                dU = np.einsum('ijk,jkr,kir->ijr', C - T, V_exp, W_exp)
                dV = np.einsum('ijk,ijr,kir->jkr', C - T, U_exp, W_exp)
                dW = np.einsum('ijk,ijr,jkr->kir', C - T, U_exp, V_exp)
                
                lr = 0.01
                U_exp -= lr * dU
                V_exp -= lr * dV
                W_exp -= lr * dW
                
                # Re-normalize?
                # U = U_exp.reshape(-1, r)
                # V = V_exp.reshape(-1, r)
                # W = W_exp.reshape(-1, r)
                
                U = U_exp.reshape(-1, r)
                V = V_exp.reshape(-1, r)
                W = W_exp.reshape(-1, r)
                
            if err < best_err:
                best_err = err
                best_rank = r
                best_U, best_V, best_W = U, V, W
                
        if best_rank == r:
            break
            
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100.0
    
    # Try to find a decomposition for 2x2x6
    # If found, use it for 2x4x6
    rank_sub, U_sub, V_sub, W_sub = _construct_sub_decomposition(2, 2, 6, 12)
    
    if rank_sub is None:
        # Fallback to previous rank 42
        # ... (reuse code from seed)
        pass
        
    # Construct full 2x4x6
    R = rank_sub * 2
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Block 1: j in 0,1
    for i in range(2):
        for j in range(2):
            for r in range(rank_sub):
                U[i*4 + j, r] = U_sub[i*2 + j, r]
                V[j*6 + k, r] = V_sub[j*6 + k, r] # Wait, k is 0..5
                # V_sub shape (12, R_sub). Index j*6+k.
                # W_sub shape (12, R_sub). Index k*2+i.
                
    # This logic is messy. Let's do it cleanly.
    
    # U_sub[i*2+j, r]
    # V_sub[j*6+k, r]
    # W_sub[k*2+i, r]
    
    for i in range(2):
        for j in range(2):
            for k in range(6):
                for r in range(rank_sub):
                    # Block 1
                    U[i*4 + j, r] = U_sub[i*2 + j, r]
                    V[j*6 + k, r] = V_sub[j*6 + k, r]
                    W[k*2 + i, r] = W_sub[k*2 + i, r]
                    
                    # Block 2 (j+2)
                    U[i*4 + (j+2), r + rank_sub] = U_sub[i*2 + j, r]
                    V[(j+2)*6 + k, r + rank_sub] = V_sub[j*6 + k, r]
                    W[k*2 + i, r + rank_sub] = W_sub[k*2 + i, r]
                    
    # Verify
    # ...
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
