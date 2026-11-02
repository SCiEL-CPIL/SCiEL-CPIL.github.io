import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if this term contributes to C[i, k]
    # C[i, k] = sum_j A[i, j] B[j, k]
    # T[i, j, k] is 1 for all i, j, k in the standard basis.
    T = np.ones((n, m, p), dtype=float)
    
    # Schoolbook fallback
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
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def objective_and_grad(params, r):
        """Compute objective and analytical gradient for tensor decomposition."""
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r : n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r:]
        
        U_t = U_flat.reshape(n, m, r)
        V_t = V_flat.reshape(m, p, r)
        W_t = W_flat.reshape(p, n, r)
        
        # Compute prod[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
        
        # Error
        error = 0.5 * np.sum((prod - T)**2)
        
        # Gradient computation
        # d_error/d_prod[i,j,k] = (prod[i,j,k] - T[i,j,k])
        dprod = prod - T
        
        # d_prod/d_U[i,j,c] = V[j,k,c] * W[k,i,c] (sum over k)
        dU = np.einsum('ijk,jkc->ijc', dprod, V_t, optimize=True)
        dU = np.einsum('ijc,kic->ijc', dU, W_t, optimize=True)
        
        # d_prod/d_V[j,k,c] = U[i,j,c] * W[k,i,c] (sum over i)
        dV = np.einsum('ijk,ijc->jkc', dprod, U_t, optimize=True)
        dV = np.einsum('jkc,kic->jkc', dV, W_t, optimize=True)
        
        # d_prod/d_W[k,i,c] = U[i,j,c] * V[j,k,c] (sum over j)
        dW = np.einsum('ijk,ijc->kic', dprod, U_t, optimize=True)
        dW = np.einsum('kic,jkc->kic', dW, V_t, optimize=True)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return error, grad
    
    best_result = None
    best_rank = rank_school
    
    # Try ranks from low to high
    for r in range(30, 56):
        if time.time() > deadline:
            break
        
        # Multiple restarts with diverse initialization strategies
        for restart in range(8):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Small random
                U_init = np.random.randn(n * m, r) * 0.1
                V_init = np.random.randn(m * p, r) * 0.1
                W_init = np.random.randn(p * n, r) * 0.1
            elif restart % 3 == 1:
                # Sparse-like random with some structure
                U_init = np.random.randn(n * m, r) * 0.05
                V_init = np.random.randn(m * p, r) * 0.05
                W_init = np.random.randn(p * n, r) * 0.05
                # Add some near-integer values
                for _ in range(r // 3):
                    U_init[np.random.randint(n*m), np.random.randint(r)] = np.random.choice([-1, 1]) * 0.5
                    V_init[np.random.randint(m*p), np.random.randint(r)] = np.random.choice([-1, 1]) * 0.5
                    W_init[np.random.randint(p*n), np.random.randint(r)] = np.random.choice([-1, 1]) * 0.5
            else:
                # Mixed scale
                U_init = np.random.randn(n * m, r) * 0.2
                V_init = np.random.randn(m * p, r) * 0.2
                W_init = np.random.randn(p * n, r) * 0.2
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Phase 1: L-BFGS-B with analytical gradient
            try:
                res1 = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    lambda x: objective_and_grad(x, r)[1],
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res1.fun < 1e-4:  # Good enough to try polishing
                    # Phase 2: Polish with tighter tolerance
                    res2 = minimize(
                        lambda x: objective_and_grad(x, r)[0],
                        lambda x: objective_and_grad(x, r)[1],
                        res1.x,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
                    )
                    
                    if res2.fun < 1e-8:
                        # Snap to half-integers
                        U_sol = res2.x[:n*m*r].reshape(n, m, r)
                        V_sol = res2.x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                        W_sol = res2.x[n*m*r + m*p*r:].reshape(p, n, r)
                        
                        U_snap = np.round(U_sol * 2) / 2
                        V_snap = np.round(V_sol * 2) / 2
                        W_snap = np.round(W_sol * 2) / 2
                        
                        # Verify exactness
                        prod_snap = np.einsum('ijc,jkc,kic->ijk', U_snap, V_snap, W_snap, optimize=True)
                        if np.allclose(prod_snap, T, atol=1e-9):
                            if r < best_rank:
                                best_rank = r
                                best_result = {
                                    "U": U_snap.reshape(n*m, r).tolist(),
                                    "V": V_snap.reshape(m*p, r).tolist(),
                                    "W": W_snap.reshape(p*n, r).tolist()
                                }
                                break  # Found solution at this rank, no need for more restarts
            except Exception:
                pass
        
        # If we found a solution at this rank, we can stop
        if best_result is not None and best_rank == r:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
