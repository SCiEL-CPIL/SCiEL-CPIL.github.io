import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 5
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-6):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(S, T, atol=tol)
    
    def solve_for_U(V, W, T, r):
        """Solve for U given V, W: U[i*m+j, r] = sum_k T[i,j,k] * V[j*p+k,r] * W[k*n+i,r] / denom"""
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                for r_idx in range(r):
                    # U[i*m+j, r_idx] * sum_k V[j*p+k, r_idx] * W[k*n+i, r_idx] = T[i,j]
                    # But T[i,j,k] = 1 for all k, so:
                    # U[i*m+j, r_idx] * sum_k V[j*p+k, r_idx] * W[k*n+i, r_idx] = sum_k 1 = p
                    # Wait, that's not right. Let me reconsider.
                    # The tensor equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
                    # So for fixed (i,j) and r: we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all k
                    # This is a system: for each k, sum_r U[i,j,r] * (V[j,k,r]*W[k,i,r]) = 1
                    # This is linear in U[i,j,:]: A[i,j] @ U[i,j,:] = ones(p)
                    # where A[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
                    pass
        
        # Actually, let's solve the full system for U given V, W
        # We have n*m*p equations: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Unknowns: n*m*r values of U
        # This is a linear system!
        
        # Build the linear system A @ U_flat = b
        # A has shape (n*m*p, n*m*r)
        # For equation (i,j,k): A[eq_idx, (i*m+j)*r + r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
        
        A = np.zeros((n * m * p, n * m * r))
        b = np.ones(n * m * p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        A[eq, (i * m + j) * r + r_idx] = V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    eq += 1
        
        # Solve the least squares problem
        U_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        return U_flat.reshape(n * m, r)
    
    def solve_for_V(U, W, T, r):
        """Solve for V given U, W"""
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        A[eq, (j * p + k) * r + r_idx] = U[i * m + j, r_idx] * W[k * n + i, r_idx]
                    eq += 1
        
        V_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        return V_flat.reshape(m * p, r)
    
    def solve_for_W(U, V, T, r):
        """Solve for W given U, V"""
        A = np.zeros((n * m * p, p * n * r))
        b = np.ones(n * m * p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        A[eq, (k * n + i) * r + r_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                    eq += 1
        
        W_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        return W_flat.reshape(p * n, r)
    
    def alternating_minimization(n, m, p, r, max_iters=50, tol=1e-10, seed=42):
        """Alternating minimization for fixed rank r"""
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to avoid trivial scaling
        for r_idx in range(r):
            norm_U = np.linalg.norm(U[:, r_idx])
            norm_V = np.linalg.norm(V[:, r_idx])
            norm_W = np.linalg.norm(W[:, r_idx])
            total = norm_U * norm_V * norm_W
            if total > 1e-10:
                U[:, r_idx] /= (total ** (1/3))
                V[:, r_idx] /= (total ** (1/3))
                W[:, r_idx] /= (total ** (1/3))
        
        prev_cost = np.inf
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve for U
            U_new = solve_for_U(V, W, T, r)
            # Solve for V
            V_new = solve_for_V(U_new, W, T, r)
            # Solve for W
            W_new = solve_for_W(U_new, V_new, T, r)
            
            # Compute cost
            U_r = U_new.reshape(n, m, r)
            V_r = V_new.reshape(m, p, r)
            W_r = W_new.reshape(p, n, r)
            S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            cost = np.sum((S - T) ** 2)
            
            if cost < prev_cost * (1 - 1e-6) or iteration == 0:
                U, V, W = U_new, V_new, W_new
                prev_cost = cost
            
            if cost < tol:
                break
        
        return U, V, W, prev_cost
    
    def try_snap_half_integer(U, V, W, n, m, p, r):
        """Try to snap to half-integers and verify"""
        # Try different scaling factors
        for scale in [0.5, 1.0, 2.0, 0.25, 0.75]:
            U_try = np.round(U * scale) / scale
            V_try = np.round(V * scale) / scale
            W_try = np.round(W * scale) / scale
            
            if verify(U_try, V_try, W_try, tol=1e-6):
                return U_try, V_try, W_try, True
        
        # Try integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int, tol=1e-6):
            return U_int, V_int, W_int, True
        
        return None, None, None, False
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But for matrix multiplication, the true lower bound is higher
    # Strassen gives ~n^2.807 for square matrices
    # For 3x4x5, let's try from 20 up
    
    np.random.seed(42)
    
    found_better = False
    for target_rank in range(20, 60):
        if time.time() > deadline - 5:
            break
        
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            U_opt, V_opt, W_opt, cost = alternating_minimization(
                n, m, p, target_rank, 
                max_iters=30, tol=1e-12, 
                seed=42 + restart
            )
            
            if cost < 1e-8:
                # Try to snap to half-integers
                U_snap, V_snap, W_snap, success = try_snap_half_integer(
                    U_opt, V_opt, W_opt, n, m, p, target_rank
                )
                
                if success:
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    found = True
                    found_better = True
                    break
                
                # If snapping fails, check if the raw solution is close enough
                if verify(U_opt, V_opt, W_opt, tol=1e-8):
                    # Try harder to snap
                    # Try with different tolerances
                    for snap_tol in [1e-4, 1e-3, 5e-3, 0.01]:
                        U_try = np.where(np.abs(U_opt * 2 - np.round(U_opt * 2)) < snap_tol, 
                                        np.round(U_opt * 2) / 2, U_opt)
                        V_try = np.where(np.abs(V_opt * 2 - np.round(V_opt * 2)) < snap_tol, 
                                        np.round(V_opt * 2) / 2, V_opt)
                        W_try = np.where(np.abs(W_opt * 2 - np.round(W_opt * 2)) < snap_tol, 
                                        np.round(W_opt * 2) / 2, W_opt)
                        if verify(U_try, V_try, W_try, tol=1e-6):
                            best_U, best_V, best_W = U_try, V_try, W_try
                            best_rank = target_rank
                            found = True
                            found_better = True
                            break
                    
                    if found:
                        break
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
