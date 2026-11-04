import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    schoolbook_rank = n * m * p  # 60
    
    # 1. Build Schoolbook Solution
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
    
    # Build Target Tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    def verify_decomposition(U_mat, V_mat, W_mat):
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = np.abs(recon - T)
        max_diff = np.max(diff)
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
        diff = recon - T
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return objective(params, R), grad

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to reduce rank using optimization
    # We try ranks 59 down to 50. If we find one, we return it.
    # This is a heuristic; it might not find the global minimum but might find a local one < 60.
    
    for R in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        # Initialize with first R columns of schoolbook
        U_init = U_sb[:, :R].copy()
        V_init = V_sb[:, :R].copy()
        W_init = W_sb[:, :R].copy()
        
        # Add small random noise to help escape local minima
        np.random.seed(42 + R)
        U_init += np.random.randn(n*m, R) * 0.1
        V_init += np.random.randn(m*p, R) * 0.1
        W_init += np.random.randn(p*n, R) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(
                lambda p: objective(p, R),
                params_init,
                jac=lambda p: objective_grad(p, R)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < 1e-6:
                U_mat = result.x[:n*m*R].reshape(n*m, R)
                V_mat = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_mat = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                is_valid, max_diff = verify_decomposition(U_mat, V_mat, W_mat)
                if is_valid:
                    best_rank = R
                    best_U, best_V, best_W = U_mat, V_mat, W_mat
                    # Found a valid lower rank, we can stop or continue to find even lower
                    # Given time constraints, stopping at the first found lower rank is safer
                    break
                    
        except Exception:
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
