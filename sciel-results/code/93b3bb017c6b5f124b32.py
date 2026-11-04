import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1

    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank_found = schoolbook_rank

    def compute_error(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - T_approx
        return np.sum(diff**2)

    def check_exact(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-10)

    def discrete_local_search(U, V, W, max_iters=5000, time_limit=5.0):
        """
        Perform a discrete local search on the half-integer grid to minimize error.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        current_error = compute_error(U, V, W)
        if current_error < 1e-10:
            return U, V, W, 0.0

        # Simulated Annealing parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.99
        current_T = T_start
        
        start_time = time.time()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random variable to change
            # U shape (9, R), V shape (15, R), W shape (15, R)
            # Total variables: 9R + 15R + 15R = 39R
            # We pick a random matrix and a random element
            mat_choice = np.random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                r_idx, c_idx = np.random.randint(0, U.shape[0]), np.random.randint(0, U.shape[1])
                old_val = U[r_idx, c_idx]
            elif mat_choice == 'V':
                r_idx, c_idx = np.random.randint(0, V.shape[0]), np.random.randint(0, V.shape[1])
                old_val = V[r_idx, c_idx]
            else:
                r_idx, c_idx = np.random.randint(0, W.shape[0]), np.random.randint(0, W.shape[1])
                old_val = W[r_idx, c_idx]
                
            # Choose a new value from neighbors (half-integers)
            # Neighbors: old_val +/- 0.5, +/- 1.0
            # To keep values bounded, we can restrict to a range, e.g., [-5, 5]
            # But for now, just +/- 0.5
            delta = np.random.choice([-0.5, 0.5])
            new_val = old_val + delta
            
            # Apply change
            if mat_choice == 'U':
                U[r_idx, c_idx] = new_val
            elif mat_choice == 'V':
                V[r_idx, c_idx] = new_val
            else:
                W[r_idx, c_idx] = new_val
                
            new_error = compute_error(U, V, W)
            
            # Acceptance criterion
            if new_error < current_error or np.random.rand() < np.exp(-(new_error - current_error) / current_T):
                current_error = new_error
            else:
                # Revert
                if mat_choice == 'U':
                    U[r_idx, c_idx] = old_val
                elif mat_choice == 'V':
                    V[r_idx, c_idx] = old_val
                else:
                    W[r_idx, c_idx] = old_val
                    
            current_T *= alpha
            if current_T < T_end:
                current_T = T_end
                
            if current_error < 1e-10:
                return U, V, W, current_error

        return U, V, W, current_error

    def objective_and_grad(params, target_rank):
        U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)

        T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        diff = T - T_approx
        loss = np.sum(diff**2)

        grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
        grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
        grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)

        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    lower_bound = max(n*m, m*p, p*n)

    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break

        max_restarts = 10 if target_rank < 25 else 5

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Initialization
            if restart == 0:
                U_init = U_sch[:, :target_rank].copy()
                V_init = V_sch[:, :target_rank].copy()
                W_init = W_sch[:, :target_rank].copy()
                U_init += np.random.randn(*U_init.shape) * 0.5
                V_init += np.random.randn(*V_init.shape) * 0.5
                W_init += np.random.randn(*W_init.shape) * 0.5
            else:
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5

            # Continuous Optimization
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])

            try:
                result = minimize(objective_and_grad, params_init, method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-5})

                U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Discrete Refinement
                U_d, V_d, W_d, err = discrete_local_search(U_r, V_r, W_d, max_iters=2000, time_limit=3.0)
                
                if err < 1e-10:
                    return {"U": U_d.tolist(), "V": V_d.tolist(), "W": W_d.tolist()}
                    
            except Exception:
                pass

            # If continuous failed, try discrete search directly from random init
            U_d, V_d, W_d, err = discrete_local_search(U_init, V_init, W_init, max_iters=2000, time_limit=3.0)
            if err < 1e-10:
                return {"U": U_d.tolist(), "V": V_d.tolist(), "W": W_d.tolist()}

    return best_result
# EVOLVE-BLOCK-END
