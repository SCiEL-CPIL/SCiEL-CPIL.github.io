import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook decomposition as baseline (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Precompute indices for the tensor equation
    # We need to evaluate T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and its gradient.
    
    # Flatten U, V, W into a single vector x for scipy
    # x[0 : n*m*R] is U
    # x[n*m*R : n*m*R + m*p*R] is V
    # x[n*m*R + m*p*R : ...] is W
    
    def unpack(x, R):
        U = x[0:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R : n*m*R + m*p*R + p*n*R].reshape(p*n, R)
        return U, V, W

    def compute_tensor_and_grad(x, R):
        U, V, W = unpack(x, R)
        T = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can compute this efficiently.
        # For each r, U_r is (n*m), V_r is (m*p), W_r is (p*n).
        # T += U_r.reshape(n,m) * V_r.reshape(m,p) * W_r.reshape(p,n) ?
        # No, the indices are coupled.
        # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's loop over r. R is ~45. n*m*p is 60.
        # Total ops ~ 45 * 60 * 3 = 8100. Very fast.
        
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # u_r[i*m+j], v_r[j*p+k], w_r[k*n+i]
            # We need to form the outer product-like structure.
            # Let's just loop over i, j, k for clarity first, then optimize if needed.
            # But 60 iterations is nothing.
            
            for i in range(n):
                for j in range(m):
                    # u_val = u_r[i*m+j]
                    u_val = u_r[i*m+j]
                    if u_val == 0: continue
                    # v_val depends on k
                    # w_val depends on k
                    for k in range(p):
                        v_val = v_r[j*p+k]
                        w_val = w_r[k*n+i]
                        term = u_val * v_val * w_val
                        T[i,j,k] += term
                        # Gradient
                        # d(T[i,j,k])/d(u_val) = v_val * w_val
                        # d(T[i,j,k])/d(v_val) = u_val * w_val
                        # d(T[i,j,k])/d(w_val) = u_val * v_val
                        grad_U[i*m+j, r] += v_val * w_val
                        grad_V[j*p+k, r] += u_val * w_val
                        grad_W[k*n+i, r] += u_val * v_val
        
        # Residual
        residual = T - 1.0
        # Loss = 0.5 * sum(residual^2)
        # Gradient of Loss w.r.t T is residual.
        # We accumulated dT/dParam. We need dL/dParam = sum(dL/dT * dT/dParam)
        # dL/dT[i,j,k] = residual[i,j,k]
        
        # So we need to contract residual with the partial derivatives.
        # The code above added to grad_U the partials for each (i,j,k).
        # We need to multiply each partial by residual[i,j,k] and sum.
        # My loop above added the partials directly. I need to re-do this or modify the loop.
        
        # Let's re-calculate gradients properly.
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            for i in range(n):
                for j in range(m):
                    u_val = u_r[i*m+j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_r[j*p+k]
                        w_val = w_r[k*n+i]
                        res = residual[i,j,k]
                        
                        # dL/dU = res * v_val * w_val
                        grad_U[i*m+j, r] += res * v_val * w_val
                        # dL/dV = res * u_val * w_val
                        grad_V[j*p+k, r] += res * u_val * w_val
                        # dL/dW = res * u_val * v_val
                        grad_W[k*n+i, r] += res * u_val * v_val
                        
        # Flatten gradients
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        loss = 0.5 * np.sum(residual**2)
        return loss, grad

    # Try to find a solution with rank < 60
    # We know rank 43 is possible (33 + 10).
    # Let's try ranks from 50 down to 40.
    
    rng = np.random.default_rng(42)
    
    for target_rank in range(50, 39, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Scale: random normal
            x0 = rng.standard_normal((n*m + m*p + p*n) * target_rank) * 0.1
            
            # Optimize
            try:
                res = minimize(
                    lambda x: compute_tensor_and_grad(x, target_rank)[0],
                    x0,
                    jac=lambda x: compute_tensor_and_grad(x, target_rank)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success and res.fun < 1e-8:
                    U, V, W = unpack(res.x, target_rank)
                    # Verify
                    T = np.zeros((n, m, p))
                    for r in range(target_rank):
                        u_r = U[:, r]
                        v_r = V[:, r]
                        w_r = W[:, r]
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T[i,j,k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                    if np.allclose(T, 1.0, atol=1e-6):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
