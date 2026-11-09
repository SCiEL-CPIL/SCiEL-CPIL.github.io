import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1

    def compute_error(U, V, W, r):
        """
        Computes the error for the tensor decomposition.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        diff = 1.0 - S
        error = np.sum(diff ** 2)
        return error

    def compute_error_and_grad(U, V, W, r):
        """
        Computes the error and the gradient for the continuous relaxation.
        """
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        diff = 1.0 - S
        error = np.sum(diff ** 2)
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V_3d, W_3d)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U_3d, W_3d)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_3d, V_3d)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad_flat

    def unpack(x, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = x[:size_U].reshape(n, m, r)
        V = x[size_U:size_U+size_V].reshape(m, p, r)
        W = x[size_U+size_V:].reshape(p, n, r)
        
        return U, V, W

    def objective(x, r):
        U, V, W = unpack(x, r)
        err, _ = compute_error_and_grad(U, V, W, r)
        return err

    def jac(x, r):
        U, V, W = unpack(x, r)
        _, grad = compute_error_and_grad(U, V, W, r)
        return grad

    # Try to find a rank-16 solution
    # Known that rank 16 is minimal for 2x3x3
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Check schoolbook first
    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    # Try rank 16 with continuous optimization
    r = 16
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        scale = 1.0
        x0 = np.random.normal(0, scale, size=(n*m + m*p + p*n) * r)
        
        # Sometimes initializing with 0.5 or 1.0 helps
        if restart % 3 == 0:
            x0 = np.random.choice([0.5, 1.0, -0.5, -1.0], size=x0.shape)
        
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective, x0, args=(r,),
                jac=lambda x: jac(x, r),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_error:
                U, V, W = unpack(res.x, r)
                err = res.fun
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    
                # Try to snap to half-integers
                if err < 1e-4:
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    # Verify exactness of snapped version
                    err_snap = compute_error(U_snapped, V_snapped, W_snapped, r)
                    
                    if err_snap < 1e-9:
                        best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                        best_rank = r
                        best_error = 0.0
                        break
                        
        except Exception:
            continue
    
    # If continuous search didn't find an exact solution, try discrete local search
    if best_error > 1e-9:
        if best_U is not None:
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
            r = best_rank
        else:
            r = 16
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, r))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, r))
        
        # Simple local search
        allowed_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        max_steps = 5000
        err = compute_error(U, V, W, r)
        
        for _ in range(max_steps):
            if time.time() > deadline - 2:
                break
            if err < 1e-9:
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
            new_val = allowed_vals[random.randint(0, len(allowed_vals)-1)]
            
            if new_val == old_val:
                continue
            
            mat[i, j] = new_val
            new_err = compute_error(U, V, W, r)
            
            if new_err < err:
                err = new_err
            else:
                mat[i, j] = old_val
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U, V, W
            best_rank = r

    # Ensure we have a valid fallback
    if best_U is None:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook
        best_error = err0

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
