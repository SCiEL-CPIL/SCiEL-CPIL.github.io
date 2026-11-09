import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 18
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Precompute index structure for fast error evaluation
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k
    
    # Flatten indices for fast computation
    # For a given r, we have:
    # U_flat: (n*m*r,), V_flat: (m*p*r,), W_flat: (p*n*r,)
    # T[i,j,k] = sum_c U_flat[(i*m+j)*r+c] * V_flat[(j*p+k)*r+c] * W_flat[(k*n+i)*r+c]
    
    def fast_error(U, V, W, r):
        """Compute sum of (T[i,j,k] - 1)^2 efficiently."""
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum for clarity; for these small sizes it's fast enough
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.sum((T - 1.0) ** 2)
    
    # Schoolbook initialization
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    err0 = fast_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank_schoolbook
        best_error = err0
    
    def try_rank(r, num_restarts=30):
        nonlocal best_U, best_V, best_W, best_rank, best_error
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Start from schoolbook and zero out some columns
                U = U0[:, :r].copy()
                V = V0[:, :r].copy()
                W = W0[:, :r].copy()
                # Add small random perturbation
                U += np.random.normal(0, 0.1, U.shape)
                V += np.random.normal(0, 0.1, V.shape)
                W += np.random.normal(0, 0.1, W.shape)
            elif restart == 1:
                # Random half-integer initialization
                U = np.random.randint(-3, 4, size=(n*m, r)) * 0.5
                V = np.random.randint(-3, 4, size=(m*p, r)) * 0.5
                W = np.random.randint(-3, 4, size=(p*n, r)) * 0.5
            elif restart == 2:
                # Sparse random initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for c in range(r):
                    U[np.random.randint(0, n*m), c] = 1.0
                    V[np.random.randint(0, m*p), c] = 1.0
                    W[np.random.randint(0, p*n), c] = 1.0
            elif restart == 3:
                # Structured: try to use Strassen-like ideas
                # For 2x3 x 3x3, think of it as 2 blocks of 1x3
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # First r/3 columns: identity-like for each (i,j)
                for c in range(min(r, n*m)):
                    idx = c % (n*m)
                    i, j = divmod(idx, m)
                    U[idx, c] = 1.0
                    V[j*p + (c % p), c] = 1.0
                    W[(c % p)*n + i, c] = 1.0
            else:
                # General random
                scale = 1.0
                U = np.random.normal(0, scale, size=(n*m, r))
                V = np.random.normal(0, scale, size=(m*p, r))
                W = np.random.normal(0, scale, size=(p*n, r))
            
            # Use scipy optimize with smooth objective
            def objective(x):
                Ux = x[:n*m*r].reshape(n*m, r)
                Vx = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                Wx = x[n*m*r+m*p*r:n*m*r+m*p*r+p*n*r].reshape(p*n, r)
                return fast_error(Ux, Vx, Wx, r)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                if res.success or res.fun < 1e-6:
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r+m*p*r:n*m*r+m*p*r+p*n*r].reshape(p*n, r)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    err_snap = fast_error(U_snap, V_snap, W_snap, r)
                    
                    # Also try without snapping (in case exact solution isn't at half-integers)
                    err_opt = res.fun
                    
                    if err_snap < 1e-9:
                        if r < best_rank or (r == best_rank and err_snap < best_error):
                            best_U, best_V, best_W = U_snap, V_snap, W_snap
                            best_rank = r
                            best_error = err_snap
                            return  # Found exact solution at this rank
                    elif err_opt < 1e-9:
                        # Solution exists but not at half-integers; try to find nearby half-integer
                        # This shouldn't happen for exact decompositions, but just in case
                        pass
                    
            except Exception:
                pass
        
        # If scipy didn't find exact, try local search on half-integer grid
        if time.time() > deadline - 5:
            return
            
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            
            U = np.random.randint(-3, 4, size=(n*m, r)) * 0.5
            V = np.random.randint(-3, 4, size=(m*p, r)) * 0.5
            W = np.random.randint(-3, 4, size=(p*n, r)) * 0.5
            
            current_err = fast_error(U, V, W, r)
            
            # Local search: try changing one entry at a time
            allowed_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            no_improve = 0
            max_no_improve = 200
            
            for step in range(3000):
                if time.time() > deadline - 2:
                    break
                if current_err < 1e-9:
                    break
                if no_improve > max_no_improve:
                    break
                
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n*m, r
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m*p, r
                else:
                    mat = W
                    rows, cols = p*n, r
                
                i = random.randint(0, rows - 1)
                j = random.randint(0, cols - 1)
                old_val = mat[i, j]
                
                # Try a few candidate values
                candidates = [v for v in allowed_vals if abs(v - old_val) > 1e-9]
                if not candidates:
                    continue
                new_val = random.choice(candidates)
                
                mat[i, j] = new_val
                new_err = fast_error(U, V, W, r)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve = 0
                else:
                    mat[i, j] = old_val
                    no_improve += 1
            
            if current_err < 1e-9:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                final_err = fast_error(U_snap, V_snap, W_snap, r)
                if final_err < 1e-9:
                    if r < best_rank or (r == best_rank and final_err < best_error):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = r
                        best_error = final_err
                        return
    
    # Try rank 16 first (known minimal for this problem)
    if time.time() < deadline - 15:
        try_rank(16, num_restarts=20)
    
    # If rank 16 not found, try rank 17
    if best_rank > 17 and time.time() < deadline - 10:
        try_rank(17, num_restarts=10)
    
    # If still not found, try rank 18 (schoolbook) as fallback
    if best_U is None:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0
    
    # Ensure entries are half-integers
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned
    
    U_list = clean_list(best_U.tolist())
    V_list = clean_list(best_V.tolist())
    W_list = clean_list(best_W.tolist())
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
