import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Define the target tensor T (flattened)
    # M[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute which entries are non-zero for fast error computation
    nonzero_indices = np.argwhere(M != 0)
    M_nonzero = M[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
    
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook (rank 27)
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    from scipy.optimize import minimize
    
    def compute_error(U, V, W):
        """Compute error efficiently using only non-zero entries of M."""
        # M_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Only need to check non-zero entries of M
        a_idx = nonzero_indices[:, 0]
        b_idx = nonzero_indices[:, 1]
        c_idx = nonzero_indices[:, 2]
        
        # Compute M_approx at non-zero positions
        # M_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        U_sub = U[a_idx, :]  # (num_nonzero, R)
        V_sub = V[b_idx, :]  # (num_nonzero, R)
        W_sub = W[c_idx, :]  # (num_nonzero, R)
        
        # Element-wise product and sum over r
        M_approx_nonzero = np.sum(U_sub * V_sub * W_sub, axis=1)
        
        # Error: sum of (M_approx - M)^2 at non-zero positions
        err = np.sum((M_approx_nonzero - M_nonzero)**2)
        
        # Also need to ensure zero positions remain zero (or close)
        # For efficiency, skip this check during optimization, verify at the end
        return err
    
    def error_fn(params, R):
        size_u = n*m * R
        size_v = m*p * R
        size_w = p*n * R
        
        U = params[:size_u].reshape(n*m, R)
        V = params[size_u:size_u+size_v].reshape(m*p, R)
        W = params[size_u+size_v:].reshape(p*n, R)
        
        return compute_error(U, V, W)
    
    def verify_solution(U, V, W):
        """Verify that the solution exactly reconstructs M."""
        M_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(M_approx, M, atol=1e-9)
    
    def polish_solution(U, V, W, R):
        """High-precision polish pass."""
        size_u = n*m * R
        size_v = m*p * R
        size_w = p*n * R
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use tighter tolerance and more iterations
        res = minimize(error_fn, x0, args=(R,), method='L-BFGS-B', 
                      options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if res.fun < 1e-10:
            U_sol = res.x[:size_u].reshape(n*m, R)
            V_sol = res.x[size_u:size_u+size_v].reshape(m*p, R)
            W_sol = res.x[size_u+size_v:].reshape(p*n, R)
            
            # Round to nearest half-integer
            U_int = np.round(U_sol * 2) / 2
            V_int = np.round(V_sol * 2) / 2
            W_int = np.round(W_sol * 2) / 2
            
            if verify_solution(U_int, V_int, W_int):
                return U_int, V_int, W_int, True
        
        return U, V, W, False
    
    # Try ranks from 19 upward
    for R in [19, 20, 21, 22, 23, 24, 25, 26]:
        if time.time() > deadline - 20:  # Keep 20s margin
            break
        
        num_restarts = 30
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Small random
                init_U = np.random.randn(n*m, R) * 0.1
                init_V = np.random.randn(m*p, R) * 0.1
                init_W = np.random.randn(p*n, R) * 0.1
            elif restart % 3 == 1:
                # Sparse random
                init_U = np.random.choice([-1, 0, 1], size=(n*m, R)) * 0.5
                init_V = np.random.choice([-1, 0, 1], size=(m*p, R)) * 0.5
                init_W = np.random.choice([-1, 0, 1], size=(p*n, R)) * 0.5
            else:
                # Mixed: start from schoolbook and add noise
                if R <= 27:
                    # Take first R columns of schoolbook
                    init_U = U_school[:, :R].copy()
                    init_V = V_school[:, :R].copy()
                    init_W = W_school[:, :R].copy()
                    # Add small noise
                    init_U += np.random.randn(n*m, R) * 0.01
                    init_V += np.random.randn(m*p, R) * 0.01
                    init_W += np.random.randn(p*n, R) * 0.01
                else:
                    init_U = np.random.randn(n*m, R) * 0.1
                    init_V = np.random.randn(m*p, R) * 0.1
                    init_W = np.random.randn(p*n, R) * 0.1
            
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                res = minimize(error_fn, x0, args=(R,), method='L-BFGS-B', 
                              options={'maxiter': 2000, 'ftol': 1e-12})
                
                if res.fun < 1e-8:
                    size_u = n*m * R
                    size_v = m*p * R
                    size_w = p*n * R
                    
                    U_sol = res.x[:size_u].reshape(n*m, R)
                    V_sol = res.x[size_u:size_u+size_v].reshape(m*p, R)
                    W_sol = res.x[size_u+size_v:].reshape(p*n, R)
                    
                    # Try to round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_int, V_int, W_int):
                        # Found valid solution!
                        best_rank = R
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        found = True
                        break
                    
                    # Try polish pass
                    U_pol, V_pol, W_pol, success = polish_solution(U_sol, V_sol, W_sol, R)
                    if success:
                        best_rank = R
                        best_U = U_pol
                        best_V = V_pol
                        best_W = W_pol
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to lists of lists as required
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
