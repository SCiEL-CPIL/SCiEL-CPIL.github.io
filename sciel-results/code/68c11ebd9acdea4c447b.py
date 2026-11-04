"""
Improved tensor rank minimization for 3x3 * 3x7 matrix multiplication.
Uses alternating minimization, structured initialization, and final polishing.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    hard_limit = 115.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    def time_left():
        return hard_limit - (time.time() - start_time)
    
    def compute_tensor(U, V, W, r):
        """Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]"""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return T
    
    def compute_tensor_and_grad(params, r):
        """
        Computes the tensor T and the gradient of the squared error ||T - 1||^2.
        params: flattened array [U_flat, V_flat, W_flat]
        """
        U = params[0:r*n*m].reshape(n * m, r)
        V = params[r*n*m:2*r*n*m].reshape(m * p, r)
        W = params[2*r*n*m:3*r*n*m].reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = T - 1.0
        obj = np.sum(err * err)
        
        # Gradients: d(obj)/dU[i,j,c] = 2 * sum_{k} err[i,j,k] * V[j,p,k] * W[k,n,i]
        # But V is indexed as V[j*p+k, c], W as W[k*n+i, c]
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        grad_W_3d = np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
        
        grad = 2.0 * np.concatenate([
            grad_U_3d.flatten(), 
            grad_V_3d.flatten(), 
            grad_W_3d.flatten()
        ])
        
        return obj, grad
    
    def objective_only(params, r):
        obj, _ = compute_tensor_and_grad(params, r)
        return obj
    
    def verify_solution(U, V, W, r):
        """Check if the solution is exact (within tolerance)"""
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def snap_to_half_integers(U, V, W, r):
        """Try to snap to half-integer values and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True
        
        # Try snapping to integers
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        
        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True
        
        return U, V, W, False
    
    # Schoolbook Solution
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1
    
    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)
    best_obj = float('inf')
    
    # Known lower bound for 3x3 * 3x7 is around 35-40 based on literature
    min_rank_to_try = 30
    max_rank_to_try = rank_schoolbook - 1
    
    def alternating_minimize(U, V, W, r, max_iter=200, tol=1e-12):
        """
        Alternating minimization: fix two matrices, solve for the third.
        For fixed V, W: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        This is a linear system in U.
        """
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for it in range(max_iter):
            if time_left() < 3:
                break
            
            # Fix V, W -> solve for U
            # T[i,j,k] = sum_c U[i*m+j,c] * A[i,j,k,c] where A[i,j,k,c] = V[j*p+k,c]*W[k*n+i,c]
            # For each (i,j): sum_c U[i*m+j,c] * A[i,j,k,c] = 1 for all k
            # This is: U_row @ A_col = 1, where U_row is (1,r), A_col is (r,p)
            # Actually: for fixed i,j, we have p equations in r unknowns
            # Let's set up: for each i,j, solve min ||U[i*m+j,:] @ A_ij - 1||^2
            # where A_ij[k,c] = V[j*p+k,c] * W[k*n+i,c]
            
            U_new = np.zeros_like(U_curr)
            for i in range(n):
                for j in range(m):
                    # A_ij shape: (p, r)
                    A_ij = V_curr[j*p:(j+1)*p, :] * W_curr[:, i]  # broadcasting: (p,r) * (p,) -> (p,r)
                    # Solve: min ||U_row @ A_ij.T - 1||^2
                    # This is: A_ij @ U_row.T = 1
                    # Least squares: U_row.T = (A_ij.T @ A_ij)^{-1} A_ij.T @ 1
                    try:
                        # Use lstsq for stability
                        U_row = np.linalg.lstsq(A_ij.T, np.ones(p), rcond=None)[0]
                        U_new[i*m+j, :] = U_row
                    except:
                        U_new[i*m+j, :] = U_curr[i*m+j, :]
            
            # Fix U, W -> solve for V
            V_new = np.zeros_like(V_curr)
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed j,k: sum_c V_row * (U[:,j] * W[k*n:(k+1)*n, i]) = 1 for all i
                    # B_ik[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    B_ik = U_curr[:, j] * W_curr[k*n:(k+1)*n, i]  # (n, r)
                    try:
                        V_row = np.linalg.lstsq(B_ik.T, np.ones(n), rcond=None)[0]
                        V_new[j*p+k, :] = V_row
                    except:
                        V_new[j*p+k, :] = V_curr[j*p+k, :]
            
            # Fix U, V -> solve for W
            W_new = np.zeros_like(W_curr)
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed k,i: sum_c W_row * (U[i*m:(i+1)*m, :] * V[j*p+k,:]) = 1 for all j
                    # C_jk[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    C_jk = U_curr[i*m:(i+1)*m, :] * V_curr[:, k]  # (m, r)
                    try:
                        W_row = np.linalg.lstsq(C_jk.T, np.ones(m), rcond=None)[0]
                        W_new[k*n+i, :] = W_row
                    except:
                        W_new[k*n+i, :] = W_curr[k*n+i, :]
            
            # Check convergence
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            T_curr = compute_tensor(U_curr, V_curr, W_curr, r)
            err = np.sum((T_curr - 1.0)**2)
            
            if err < tol:
                break
        
        T_final = compute_tensor(U_curr, V_curr, W_curr, r)
        obj_final = np.sum((T_final - 1.0)**2)
        
        return U_curr, V_curr, W_curr, obj_final
    
    def optimize_with_lbfgs(U0, V0, W0, r, maxiter=1000):
        """Optimize using L-BFGS-B"""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(
                lambda x: objective_only(x, r),
                params0,
                jac=lambda x: compute_tensor_and_grad(x, r)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            params_opt = res.x
            U_opt = params_opt[0:r*n*m].reshape(n*m, r)
            V_opt = params_opt[r*n*m:2*r*n*m].reshape(m*p, r)
            W_opt = params_opt[2*r*n*m:3*r*n*m].reshape(p*n, r)
            
            obj_val = objective_only(params_opt, r)
            return U_opt, V_opt, W_opt, obj_val
            
        except Exception:
            return U0, V0, W0, float('inf')
    
    def structured_init(r, seed_val):
        """
        Create a structured initialization that respects the bilinear form.
        Start from a rank-r approximation of the schoolbook solution.
        """
        np.random.seed(seed_val)
        
        # Create random projections to get rank r from schoolbook
        U_proj = np.random.randn(n*m, r)
        V_proj = np.random.randn(m*p, r)
        W_proj = np.random.randn(p*n, r)
        
        # Scale to have reasonable magnitudes
        U0 = U_proj * 0.5
        V0 = V_proj * 0.5
        W0 = W_proj * 0.5
        
        return U0, V0, W0
    
    def sparse_init(r, seed_val):
        """Sparse initialization with random non-zero entries"""
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        U0 = np.zeros((n*m, r))
        V0 = np.zeros((m*p, r))
        W0 = np.zeros((p*n, r))
        
        # Each row has 1-2 non-zero entries
        for i in range(n*m):
            num_nonzero = random.randint(1, 2)
            cols = random.sample(range(r), num_nonzero)
            for c in cols:
                U0[i, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        for j in range(m*p):
            num_nonzero = random.randint(1, 2)
            cols = random.sample(range(r), num_nonzero)
            for c in cols:
                V0[j, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        for k in range(p*n):
            num_nonzero = random.randint(1, 2)
            cols = random.sample(range(r), num_nonzero)
            for c in cols:
                W0[k, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        return U0, V0, W0
    
    # Try different ranks from high to low
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time_left() < 5:
            break
            
        found_valid = False
        num_restarts = 4 if r > 45 else 6
        
        for trial in range(num_restarts):
            if time_left() < 5:
                break
            
            seed_val = trial * 1000 + r * 7
            
            # Different initialization strategies
            if trial == 0:
                U0, V0, W0 = structured_init(r, seed_val)
            elif trial == 1:
                U0, V0, W0 = sparse_init(r, seed_val)
            elif trial == 2:
                # Random small values + bias
                np.random.seed(seed_val)
                U0 = np.random.randn(n*m, r) * 0.1 + 0.5
                V0 = np.random.randn(m*p, r) * 0.1 + 0.5
                W0 = np.random.randn(p*n, r) * 0.1 + 0.5
            else:
                # Very small random values
                np.random.seed(seed_val + 100)
                U0 = np.random.randn(n*m, r) * 0.01
                V0 = np.random.randn(m*p, r) * 0.01
                W0 = np.random.randn(p*n, r) * 0.01
            
            # Try alternating minimization first (faster, more structured)
            U_alt, V_alt, W_alt, obj_alt = alternating_minimize(U0, V0, W0, r, max_iter=100, tol=1e-14)
            
            # If alternating minimization got close, try L-BFGS for polishing
            if obj_alt < 1e-4:
                U_opt, V_opt, W_opt, obj_val = optimize_with_lbfgs(U_alt, V_alt, W_alt, r, maxiter=500)
            else:
                U_opt, V_opt, W_opt, obj_val = U_alt, V_alt, W_alt, obj_alt
            
            # Check if we found a very good solution
            if obj_val < 1e-6:
                U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(U_opt, V_opt, W_opt, r)
                
                if is_valid:
                    best_rank = r
                    best_solution = (U_snap, V_snap, W_snap)
                    found_valid = True
                    break
            
            # If we haven't found a valid solution, update best if this is better
            if obj_val < best_obj:
                best_obj = obj_val
        
        if found_valid:
            continue  # Try next lower rank
    
    U_res, V_res, W_res = best_solution
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
