"""Rewritten program: systematic search for low-rank tensor decomposition of 3x4x5 matrix multiplication."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    schoolbook_rank = n * m * p  # 60
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        U = np.zeros((nm, schoolbook_rank))
        V = np.zeros((mp, schoolbook_rank))
        W = np.zeros((pn, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U_mat, V_mat, W_mat, tol=1e-6):
        """Verify exact decomposition: sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        max_diff = np.max(np.abs(recon - T))
        return max_diff < tol, max_diff
    
    def objective_and_grad(params, R):
        """Compute SSE objective and gradient for rank-R decomposition."""
        U_size = nm * R
        V_size = mp * R
        U_mat = params[:U_size].reshape(nm, R)
        V_mat = params[U_size:U_size+V_size].reshape(mp, R)
        W_mat = params[U_size+V_size:].reshape(pn, R)
        
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = recon - T
        error = np.sum(diff * diff)
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        return error, grad
    
    def try_rank(R, num_restarts=3, max_iter=200, tol=1e-6):
        """Try to find a valid decomposition of rank R. Returns (success, U, V, W)."""
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return False, None, None
            
            # Initialize: for high ranks use schoolbook-like init, for lower use random
            if R >= schoolbook_rank - 5:
                # Schoolbook init: fill first min(R, 60) columns
                U_mat = np.zeros((nm, R))
                V_mat = np.zeros((mp, R))
                W_mat = np.zeros((pn, R))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < R:
                                U_mat[i*m+j, col] = 1.0
                                V_mat[j*p+k, col] = 1.0
                                W_mat[k*n+i, col] = 1.0
                            col += 1
                # Small perturbation to break symmetry
                U_mat += np.random.randn(nm, R) * 0.05
                V_mat += np.random.randn(mp, R) * 0.05
                W_mat += np.random.randn(pn, R) * 0.05
            else:
                U_mat = np.random.randn(nm, R) * 0.5
                V_mat = np.random.randn(mp, R) * 0.5
                W_mat = np.random.randn(pn, R) * 0.5
            
            params = np.concatenate([U_mat.ravel(), V_mat.ravel(), W_mat.ravel()])
            
            try:
                result = minimize(
                    lambda p: objective_and_grad(p, R)[0],
                    params,
                    jac=lambda p: objective_and_grad(p, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                
                if result.fun < tol:
                    U_r = result.x[:U_size].reshape(nm, R)
                    V_r = result.x[U_size:U_size+V_size].reshape(mp, R)
                    W_r = result.x[U_size+V_size:].reshape(pn, R)
                    valid, md = verify(U_r, V_r, W_r)
                    if valid:
                        return True, U_r, V_r, W_r
            except Exception:
                pass
        
        return False, None, None
    
    # Try ranks from 55 down to 35
    # The known lower bound for 3x4x5 is at least 3*4*5 - something
    # Strassen for 2x2x2 gives 7 instead of 8, so savings are possible
    # For 3x4x5, let's try to find the minimum
    
    # First try structured approaches for specific ranks
    # Try rank 55 (save 5 from schoolbook)
    for R in range(55, 34, -1):
        if time.time() > deadline - 15:
            break
        
        # More restarts for higher ranks (easier to find), fewer for lower
        if R >= 53:
            num_restarts = 5
            max_iter = 150
        elif R >= 48:
            num_restarts = 4
            max_iter = 200
        else:
            num_restarts = 3
            max_iter = 250
        
        success, U_r, V_r, W_r = try_rank(R, num_restarts, max_iter)
        
        if success:
            best_rank = R
            best_U, best_V, best_W = U_r, V_r, W_r
            # Found a solution, continue trying lower ranks
            continue
        else:
            # If we can't find at this rank, try one more restart with different init
            if R <= 52:
                # Try a different initialization strategy
                U_mat = np.random.randn(nm, R) * 0.3
                V_mat = np.random.randn(mp, R) * 0.3
                W_mat = np.random.randn(pn, R) * 0.3
                params = np.concatenate([U_mat.ravel(), V_mat.ravel(), W_mat.ravel()])
                try:
                    result = minimize(
                        lambda p: objective_and_grad(p, R)[0],
                        params,
                        jac=lambda p: objective_and_grad(p, R)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if result.fun < 1e-6:
                        U_r = result.x[:nm*R].reshape(nm, R)
                        V_r = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                        W_r = result.x[nm*R+mp*R:].reshape(pn, R)
                        valid, md = verify(U_r, V_r, W_r)
                        if valid:
                            best_rank = R
                            best_U, best_V, best_W = U_r, V_r, W_r
                            continue
                except Exception:
                    pass
            break  # Stop descending if we can't find at this rank
    
    # Final polish: if we found a solution, try to verify with tighter tolerance
    if best_rank < schoolbook_rank:
        valid, md = verify(best_U, best_V, best_W, tol=1e-4)
        if not valid:
            # Fall back to schoolbook
            best_rank = schoolbook_rank
            best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
