import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140
    
    # Precompute the target tensor T indices
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    a_idx = np.array([x[0] for x in nz_indices])
    b_idx = np.array([x[1] for x in nz_indices])
    c_idx = np.array([x[2] for x in nz_indices])
    
    def error_function(params, rank):
        # Unpack params
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank :]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Compute R on non-zero indices
        R_nz = np.zeros(len(nz_indices))
        for s in range(rank):
            R_nz += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
        
        # Error
        err = np.sum((R_nz - 1.0)**2)
        
        # Optional: Add a small penalty to keep values bounded?
        # L-BFGS-B handles bounds well.
        
        return err

    def gradient_function(params, rank):
        # Numerical gradient or analytical?
        # Analytical is better for speed and stability.
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank :]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        R_nz = np.zeros(len(nz_indices))
        for s in range(rank):
            R_nz += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
        
        diff = R_nz - 1.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for s in range(rank):
            # dErr/dU[a,s] = 2 * diff * V[b,s] * W[c,s]
            # Sum over all nz indices where a_idx is a
            # We can do this vectorized
            term_U = diff * V[b_idx, s] * W[c_idx, s]
            np.add.at(grad_U, (a_idx, s), term_U)
            
            term_V = diff * U[a_idx, s] * W[c_idx, s]
            np.add.at(grad_V, (b_idx, s), term_V)
            
            term_W = diff * U[a_idx, s] * V[b_idx, s]
            np.add.at(grad_W, (c_idx, s), term_W)
        
        grad_U *= 2
        grad_V *= 2
        grad_W *= 2
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def try_rank(rank, num_restarts=20, time_budget=20):
        start_time = time.time()
        best_solution = None
        best_error = np.inf
        
        for r in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            if time.time() > deadline - 5:
                return None, best_error
                
            np.random.seed(1000 + rank * 10 + r)
            
            # Initialize
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Bounds
            # Keep values in a reasonable range, e.g., [-2, 2]
            bounds = [(-2.0, 2.0)] * (n*m + m*p + p*n) * rank
            
            # Optimize
            try:
                res = minimize(
                    error_function, 
                    params_init, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    jac=gradient_function,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < best_error:
                if res.fun < best_error:
                    best_error = res.fun
                    
                # Snap to half-integers
                U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                U_half = np.round(U_opt * 2) / 2
                V_half = np.round(V_opt * 2) / 2
                W_half = np.round(W_opt * 2) / 2
                
                # Check exactness
                R_check = np.einsum('as,bs,cs->abc', U_half, V_half, W_half)
                # Only check non-zero indices for speed
                R_nz_check = R_check[a_idx, b_idx, c_idx]
                
                if np.allclose(R_nz_check, 1.0, atol=1e-9):
                    # Verify full tensor is zero elsewhere? 
                    # The decomposition might create non-zero entries elsewhere.
                    # We must ensure R_check is exactly T (0 elsewhere).
                    T_full = np.zeros((n*m, m*p, p*n))
                    T_full[a_idx, b_idx, c_idx] = 1.0
                    if np.allclose(R_check, T_full, atol=1e-9):
                        best_solution = (U_half, V_half, W_half)
                        return best_solution, best_error # Found exact, return immediately
        
        return best_solution, best_error

    # Try rank 16 first
    sol, cost = try_rank(16, num_restarts=30, time_budget=30)
    if sol is not None:
        U, V, W = sol
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
        
    # Try rank 15
    sol, cost = try_rank(15, num_restarts=50, time_budget=60)
    if sol is not None:
        U, V, W = sol
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
        
    # Fallback: schoolbook
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
