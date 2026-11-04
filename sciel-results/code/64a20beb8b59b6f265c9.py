"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # leave margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    schoolbook_rank = n * m * p  # 60
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build the full tensor as a 3D array for verification
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if U * V * W (tensor product) equals T exactly"""
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = np.abs(recon - T)
        max_diff = np.max(diff) if recon.size > 0 else 0
        return max_diff < 1e-8, max_diff

    def objective(params, R):
        U_size = n*m*R
        V_size = m*p*R
        U_flat = params[:U_size].reshape(n*m, R)
        V_flat = params[U_size:U_size+V_size].reshape(m*p, R)
        W_flat = params[U_size+V_size:].reshape(p*n, R)
        recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        error = np.sum((recon - T)**2)
        return error
    
    def objective_grad(params, R):
        U_size = n*m*R
        V_size = m*p*R
        U_mat = params[:U_size].reshape(n*m, R)
        V_mat = params[U_size:U_size+V_size].reshape(m*p, R)
        W_mat = params[U_size+V_size:].reshape(p*n, R)
        
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        diff = recon - T
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad

    # Try to find solutions for ranks below schoolbook using gradient-based optimization
    # with multiple restarts and structured initializations
    
    from scipy.optimize import minimize
    
    best_solution = None
    
    # Try ranks from 55 down to 35, with multiple restarts
    for R in range(55, 34, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random restarts with different initializations
        num_restarts = 6
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            np.random.seed(42 + restart * 1000 + R)
            
            if restart % 3 == 0:
                # Initialize near schoolbook (for higher ranks)
                U_flat = np.zeros((n*m, R))
                V_flat = np.zeros((m*p, R))
                W_flat = np.zeros((p*n, R))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < R:
                                U_flat[i*m+j, col] = 1
                                V_flat[j*p+k, col] = 1
                                W_flat[k*n+i, col] = 1
                            col += 1
                # Add small random perturbation
                U_flat += np.random.randn(n*m, R) * 0.05
                V_flat += np.random.randn(m*p, R) * 0.05
                W_flat += np.random.randn(p*n, R) * 0.05
            elif restart % 3 == 1:
                # Random initialization
                U_flat = np.random.randn(n*m, R) * 0.1
                V_flat = np.random.randn(m*p, R) * 0.1
                W_flat = np.random.randn(p*n, R) * 0.1
            else:
                # Sparse-like initialization: many zeros, some random non-zeros
                U_flat = np.zeros((n*m, R))
                V_flat = np.zeros((m*p, R))
                W_flat = np.zeros((p*n, R))
                for r in range(R):
                    # Randomly select a few non-zero entries per column
                    n_nonzero_U = np.random.randint(1, min(5, n*m) + 1)
                    n_nonzero_V = np.random.randint(1, min(5, m*p) + 1)
                    n_nonzero_W = np.random.randint(1, min(5, p*n) + 1)
                    
                    idx_U = np.random.choice(n*m, n_nonzero_U, replace=False)
                    idx_V = np.random.choice(m*p, n_nonzero_V, replace=False)
                    idx_W = np.random.choice(p*n, n_nonzero_W, replace=False)
                    
                    U_flat[idx_U, r] = np.random.choice([-1, 0.5, 1, -0.5])
                    V_flat[idx_V, r] = np.random.choice([-1, 0.5, 1, -0.5])
                    W_flat[idx_W, r] = np.random.choice([-1, 0.5, 1, -0.5])
            
            params = np.concatenate([U_flat.flatten(), V_flat.flatten(), W_flat.flatten()])
            
            try:
                result = minimize(
                    lambda p: objective(p, R),
                    params,
                    jac=lambda p: objective_grad(p, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if result.fun < 1e-8:
                    U_mat = result.x[:n*m*R].reshape(n*m, R)
                    V_mat = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_mat = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    is_valid, max_diff = verify_decomposition(U_mat, V_mat, W_mat)
                    if is_valid:
                        best_solution = (R, U_mat.copy(), V_mat.copy(), W_mat.copy())
                        best_rank = R
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            continue  # Try to go lower
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_solution is not None:
        R, U_mat, V_mat, W_mat = best_solution
        best_rank = R
    else:
        U_mat, V_mat, W_mat = U_sb, V_sb, W_sb
    
    # Convert to lists as required by the interface
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
