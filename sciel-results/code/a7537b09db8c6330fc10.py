"""Matrix multiplication 2x3x5 tensor decomposition - rank minimization."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] as a 3D array."""
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T."""
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, n*p))
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    T_true = _build_tensor(n, m, p)
    return np.max(np.abs(T_recon - T_true)) < tol

def _polish(U, V, W, T, n, m, p, max_iters=500, tol=1e-12):
    """High-precision polish pass with adaptive step size."""
    rank = U.shape[1]
    lr = 0.01
    
    for it in range(max_iters):
        # Compute residual
        recon = np.zeros((n*m, m*p, n*p))
        for c in range(rank):
            recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        residual = recon - T
        err = np.sum(residual**2)
        
        if err < tol:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for c in range(rank):
            vw = np.einsum('b,c->bc', V[:,c], W[:,c])
            uw = np.einsum('a,c->ac', U[:,c], W[:,c])
            uv = np.einsum('a,b->ab', U[:,c], V[:,c])
            dU[:,c] = np.einsum('abc,bc->a', residual, vw)
            dV[:,c] = np.einsum('abc,ac->b', residual, uw)
            dW[:,c] = np.einsum('abc,ab->c', residual, uv)
        
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 1e-15:
            step_size = min(0.1 / grad_norm, lr)
        else:
            step_size = lr
        
        U -= step_size * dU
        V -= step_size * dV
        W -= step_size * dW
        
        # Scale invariance fix
        for c in range(rank):
            norm_u = np.linalg.norm(U[:,c])
            norm_v = np.linalg.norm(V[:,c])
            norm_w = np.linalg.norm(W[:,c])
            if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                target = (norm_u * norm_v * norm_w) ** (1/3)
                U[:,c] *= target / norm_u
                V[:,c] *= target / norm_v
                W[:,c] *= target / norm_w
        
        # Adaptive learning rate adjustment
        if it % 20 == 0:
            if err > 1.0:
                lr *= 0.8
            elif err < 0.01:
                lr = min(lr * 1.2, 0.05)
    
    return U, V, W, err

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_school
    
    T = _build_tensor(n, m, p)
    
    def _try_rank(rank, max_iters=2000, n_restarts=5):
        nonlocal best_result, best_rank
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization with varied scales
            scale = 0.1 * (1 + 0.5 * random.random())
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # Gradient descent on F = 0.5 * ||sum_c u_c ⊗ v_c ⊗ w_c - T||^2
            lr = 0.01
            prev_err = float('inf')
            stagnant = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Compute residual
                recon = np.zeros((n*m, m*p, n*p))
                for c in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                residual = recon - T
                err = np.sum(residual**2)
                
                if err < 1e-10:
                    # Found near-exact decomposition, polish it
                    U, V, W, final_err = _polish(U, V, W, T, n, m, p, max_iters=300, tol=1e-13)
                    if final_err < 1e-10:
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        return True
                    # Polish didn't converge, continue
                    err = final_err
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for c in range(rank):
                    vw = np.einsum('b,c->bc', V[:,c], W[:,c])
                    uw = np.einsum('a,c->ac', U[:,c], W[:,c])
                    uv = np.einsum('a,b->ab', U[:,c], V[:,c])
                    dU[:,c] = np.einsum('abc,bc->a', residual, vw)
                    dV[:,c] = np.einsum('abc,ac->b', residual, uw)
                    dW[:,c] = np.einsum('abc,ab->c', residual, uv)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-15:
                    step_size = min(0.05 / grad_norm, lr)
                else:
                    step_size = lr
                
                U -= step_size * dU
                V -= step_size * dV
                W -= step_size * dW
                
                # Scale invariance fix: normalize columns
                for c in range(rank):
                    norm_u = np.linalg.norm(U[:,c])
                    norm_v = np.linalg.norm(V[:,c])
                    norm_w = np.linalg.norm(W[:,c])
                    if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                        target = (norm_u * norm_v * norm_w) ** (1/3)
                        U[:,c] *= target / norm_u
                        V[:,c] *= target / norm_v
                        W[:,c] *= target / norm_w
                
                # Stagnation detection
                if err > prev_err * 0.999:
                    stagnant += 1
                else:
                    stagnant = 0
                prev_err = err
                
                if stagnant > 50:
                    # Restart with perturbation
                    U += np.random.randn(n*m, rank) * 0.01
                    V += np.random.randn(m*p, rank) * 0.01
                    W += np.random.randn(p*n, rank) * 0.01
                    stagnant = 0
                
                if it % 50 == 0 and err > 1.0:
                    lr *= 0.9
                elif it % 50 == 0 and err < 0.1:
                    lr = min(lr * 1.1, 0.05)
        
        return False
    
    # Try ranks from lower bound upward
    lower_bound = n*m + m*p + n*p - n - m - p + 1  # = 22
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = _try_rank(rank, max_iters=1500, n_restarts=4)
        if success:
            break
    
    # Convert to lists
    return best_result

# EVOLVE-BLOCK-END
