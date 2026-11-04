"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_numpy(U, V, W, rank, tol=1e-9):
        """Faster verification using numpy."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half(x):
        """Snap values to nearest half-integer."""
        return np.round(2.0 * x) / 2.0
    
    def schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank
    
    def objective(params, rank):
        """
        Objective: sum of (sum_c U_ijc * V_jkc * W_kic - 1)^2 over all (i,j,k).
        params: flat vector of length (n*m + m*p + p*n) * rank
        Layout: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        """
        U = params[0:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute residual efficiently
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res += (val - 1.0) ** 2
        return res
    
    def alternating_minimization(U, V, W, rank, max_iters=100, tol=1e-12):
        """
        Alternating least squares for tensor decomposition.
        Returns (U, V, W, obj_value)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U: fix V, W
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is: A @ U[i*m+j,:] = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: fix U, W
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    # For each (i,jj): sum_c U[i*m+jj,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # A[(i,jj),c] = U[i*m+jj,c]*W[k*n+i,c], b[(i,jj)] = 1
                    A = np.zeros((n*m, rank))
                    b = np.ones(n*m)
                    for i in range(n):
                        for jj in range(m):
                            A[i*m+jj, :] = U_new[i*m+jj, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: fix U, V
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    # For each (jj,ii): sum_c U[ii*m+jj,c] * V[jj*p+k,c] * W[k*n+i,c] = 1
                    # A[(ii,jj),c] = U[ii*m+jj,c]*V[jj*p+k,c], b[(ii,jj)] = 1
                    A = np.zeros((n*m, rank))
                    b = np.ones(n*m)
                    for ii in range(n):
                        for jj in range(m):
                            A[ii*m+jj, :] = U_new[ii*m+jj, :] * V_new[jj*p+k, :]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if it % 5 == 0 or it == max_iters - 1:
                obj = objective(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), rank)
                if obj < tol:
                    break
        
        obj = objective(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), rank)
        return U, V, W, obj
    
    def try_rank_opt(rank, n_starts=10, max_iters=80):
        """
        Try to find a decomposition of given rank using optimization.
        """
        if time.time() > deadline - 20:
            return None
        
        best_result = None
        best_obj = float('inf')
        
        for start in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if start % 3 == 0:
                # Strategy 1: Structured random
                np.random.seed(42 + start * 7 + rank)
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.4:
                                U_init[i*m+j, c] = np.random.choice([-1.0, 1.0])
                    for j in range(m):
                        for k in range(p):
                            if np.random.random() < 0.4:
                                V_init[j*p+k, c] = np.random.choice([-1.0, 1.0])
                    for k in range(p):
                        for i in range(n):
                            if np.random.random() < 0.4:
                                W_init[k*n+i, c] = np.random.choice([-1.0, 1.0])
                
            elif start % 3 == 1:
                # Strategy 2: Small random values
                np.random.seed(100 + start * 13 + rank)
                U_init = np.random.randn(n*m, rank) * 0.5
                V_init = np.random.randn(m*p, rank) * 0.5
                W_init = np.random.randn(p*n, rank) * 0.5
                
            else:
                # Strategy 3: Based on schoolbook structure but compressed
                np.random.seed(200 + start * 17 + rank)
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                
                # Try to reuse patterns
                for c in range(rank):
                    # Pick random subsets
                    i_indices = np.random.choice(n, size=1, replace=False)
                    j_indices = np.random.choice(m, size=1, replace=False)
                    k_indices = np.random.choice(p, size=1, replace=False)
                    
                    for i in i_indices:
                        for j in j_indices:
                            U_init[i*m+j, c] = 1.0
                    for j in j_indices:
                        for k in k_indices:
                            V_init[j*p+k, c] = 1.0
                    for k in k_indices:
                        for i in i_indices:
                            W_init[k*n+i, c] = 1.0
                
                # Add some random perturbation
                U_init += np.random.randn(n*m, rank) * 0.1
                V_init += np.random.randn(m*p, rank) * 0.1
                W_init += np.random.randn(p*n, rank) * 0.1
            
            # Run alternating minimization
            U, V, W, obj = alternating_minimization(U_init, V_init, W_init, rank, max_iters=max_iters)
            
            if obj < best_obj:
                best_obj = obj
                best_result = (U, V, W)
            
            # If we found a very good solution, try to snap
            if obj < 1e-10:
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                    return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # If good but not perfect, do a polish pass
            if obj < 1e-6:
                # Try local optimization with scipy
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def obj_func(p):
                    return objective(p, rank)
                
                try:
                    res = minimize(obj_func, params, method='BFGS', 
                                   options={'maxiter': 50, 'gtol': 1e-10})
                    if res.fun < best_obj:
                        best_obj = res.fun
                        best_result = (
                            res.x[0:n*m*rank].reshape(n*m, rank),
                            res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank),
                            res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        )
                        
                        if res.fun < 1e-10:
                            U_snap = snap_to_half(best_result[0])
                            V_snap = snap_to_half(best_result[1])
                            W_snap = snap_to_half(best_result[2])
                            if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                except:
                    pass
        
        # Final attempt: if we have a good solution, try to snap with perturbations
        if best_result is not None and best_obj < 1e-6:
            U, V, W = best_result
            
            # Try snapping directly
            U_snap = snap_to_half(U)
            V_snap = snap_to_half(V)
            W_snap = snap_to_half(W)
            if verify_numpy(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
            
            # Try small perturbations
            for _ in range(30):
                if time.time() > deadline - 10:
                    break
                U_pert = U + np.random.randn(n*m, rank) * 1e-3
                V_pert = V + np.random.randn(m*p, rank) * 1e-3
                W_pert = W + np.random.randn(p*n, rank) * 1e-3
                U_s = snap_to_half(U_pert)
                V_s = snap_to_half(V_pert)
                W_s = snap_to_half(W_pert)
                if verify_numpy(U_s.tolist(), V_s.tolist(), W_s.tolist(), rank, tol=1e-8):
                    return (U_s.tolist(), V_s.tolist(), W_s.tolist())
            
            # Try rounding to integers as well
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify_numpy(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, tol=1e-8):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        
        return None
    
    # Try ranks from low to high
    # Based on theoretical bounds, rank should be around 24-27 for 2x3x6
    for rank in range(22, 37):
        if time.time() > deadline - 25:
            break
        
        result = try_rank_opt(rank, n_starts=8, max_iters=60)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
