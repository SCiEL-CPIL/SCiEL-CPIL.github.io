import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)

    def gradient_descent(target_rank, max_iters=10000, tol=1e-12, seed=None, init_U=None, init_V=None, init_W=None):
        """Gradient Descent with momentum for tensor decomposition."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        if init_U is None:
            U = rng.randn(n * m, target_rank) * 0.1
        else:
            U = init_U.copy()
        if init_V is None:
            V = rng.randn(m * p, target_rank) * 0.1
        else:
            V = init_V.copy()
        if init_W is None:
            W = rng.randn(p * n, target_rank) * 0.1
        else:
            W = init_W.copy()
            
        prev_err = float('inf')
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
                
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # Compute current tensor
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error_tensor = result - T
            err = np.linalg.norm(error_tensor)
            
            if err < tol:
                return U, V, W, err
            
            # Compute gradients
            # d/dU[i,j,r] = sum_{k} error[i,j,k] * V[j,k,r] * W[k,i,r]
            # d/dV[j,k,r] = sum_{i} error[i,j,k] * U[i,j,r] * W[k,i,r]
            # d/dW[k,i,r] = sum_{j} error[i,j,k] * U[i,j,r] * V[j,k,r]
            
            # Using einsum for efficiency
            grad_U = np.einsum('ijk,jkr,kir->ijr', error_tensor, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', error_tensor, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', error_tensor, U_r, V_r)
            
            # Update with momentum
            vU = momentum * vU - lr * grad_U
            U = U + vU
            
            vV = momentum * vV - lr * grad_V
            V = V + vV
            
            vW = momentum * vW - lr * grad_W
            W = W + vW
            
            if abs(err - prev_err) < 1e-15:
                # Try to reduce learning rate if stuck
                lr *= 0.5
                if lr < 1e-6:
                    break
            prev_err = err
            
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err

    # Try ranks from 14 upward
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        max_restarts = 20 if target_rank <= 15 else 10
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Use random init for most, but maybe try one init close to schoolbook if rank is close?
            # Actually, random is usually better for low rank.
            U, V, W, err = gradient_descent(target_rank, max_iters=2000, tol=1e-10, seed=restart)
            
            if U is None:
                break
                
            if err < 1e-8:
                # Polish with higher precision
                U_pol, V_pol, W_pol, err_pol = gradient_descent(target_rank, max_iters=5000, tol=1e-14, seed=restart, init_U=U, init_V=V, init_W=W)
                if U_pol is not None and err_pol < 1e-10:
                    best_U, best_V, best_W = U_pol, V_pol, W_pol
                    best_rank = target_rank
                    found = True
                    break
                elif err < 1e-10:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
                    
        if found:
            continue
        else:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
