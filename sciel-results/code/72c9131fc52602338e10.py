import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = DIMENSIONS
    
    # Precompute indices for the non-zero entries of the tensor T
    # T[i, j, j, k, i, k] = 1
    # We need to satisfy: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    num_constraints = len(indices) # n*m*p = 84

    def compute_residuals(U, V, W, R):
        """
        Computes the residuals for the non-zero entries.
        U: (R, n*m), V: (R, m*p), W: (R, p*n)
        Returns array of shape (num_constraints,)
        """
        residuals = np.zeros(num_constraints)
        for idx, (i, j, k) in enumerate(indices):
            u_val = U[:, i * m + j]
            v_val = V[:, j * p + k]
            w_val = W[:, k * n + i]
            # sum_r u_val[r] * v_val[r] * w_val[r]
            residuals[idx] = np.sum(u_val * v_val * w_val)
        return residuals - 1.0

    def objective(params, R, lam):
        """
        Objective function: Sum of squares of residuals + regularization
        """
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((R, n * m))
        V = V_flat.reshape((R, m * p))
        W = W_flat.reshape((R, p * n))
        
        res = compute_residuals(U, V, W, R)
        
        # Sum of squares of residuals
        ss_res = np.sum(res * res)
        
        # Regularization (L2) to keep values small
        reg = lam * (np.sum(U * U) + np.sum(V * V) + np.sum(W * W))
        
        return ss_res + reg

    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 45 to 83
    # Theoretical lower bound for 3x4x7 is not strictly known but 45 is a reasonable start for search
    for R in range(45, min(84, n*m*p)):
        if time.time() > deadline:
            break
            
        num_restarts = 10 if R < 60 else 5
        lam = 1e-4 # Small regularization
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            # Initialization
            if restart == 0:
                # Start from a scaled random combination of schoolbook basis
                # This is essentially a random point in the space
                x0 = np.random.randn(total_size) * 1.0
            else:
                # Random initialization
                x0 = np.random.randn(total_size) * 0.5
            
            try:
                # Use L-BFGS-B which is robust for this size
                res = minimize(
                    objective, 
                    x0, 
                    args=(R, lam), 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            # Check if we found a solution
            if res.fun < 1e-6: # Threshold for success
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_cont = U_flat.reshape((n*m, R))
                V_cont = V_flat.reshape((m*p, R))
                W_cont = W_flat.reshape((p*n, R))
                
                # Snap to half-integers
                U_snap = np.round(U_cont * 2) / 2
                V_snap = np.round(V_cont * 2) / 2
                W_snap = np.round(W_cont * 2) / 2
                
                # Verify snapped solution
                # We need to check if the snapped solution exactly satisfies the constraints
                res_snapped = compute_residuals(
                    U_snap.T, # compute_residuals expects (R, dim)
                    V_snap.T,
                    W_snap.T,
                    R
                )
                
                if np.max(np.abs(res_snapped)) < 1e-9:
                    best_rank = R
                    best_solution = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        return schoolbook_solution()
