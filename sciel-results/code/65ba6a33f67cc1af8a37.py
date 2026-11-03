import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    # 1. Construct Schoolbook Decomposition (Rank n*m*p = 48)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # 2. Try to find lower rank decompositions
    # We search from rank 47 down to a reasonable lower bound
    # Lower bound is max(n*m, m*p, p*n) = 16
    # But practically, we might not find it below 30 quickly.
    
    for rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        found = _try_find_rank_from_schoolbook(n, m, p, rank, U_sb, V_sb, W_sb, max_restarts=50, time_limit=2.0)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            break  # Found a valid one, stop searching lower ranks immediately? 
            # Actually, we want the MINIMAL rank. 
            # But if we found 47, we should try 46. 
            # However, if 47 is valid, 48 is not minimal. 
            # We should continue to lower ranks if time permits.
            # But the loop goes DOWN. So if we find 47, we break the inner loop 
            # and the outer loop continues to 46.
            # Wait, the code structure:
            # for rank in ...:
            #   found = ...
            #   if found:
            #      best = ...
            #      # Do NOT break outer loop, continue to next lower rank
            #      # But we need to update best_rank
            #      # The loop variable 'rank' decreases.
            #      # So we just continue.
            #      pass
            #   else:
            #      # try next rank
            
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_rank_from_schoolbook(n, m, p, rank, U_sb, V_sb, W_sb, max_restarts=10, time_limit=5.0):
    """
    Attempts to find a rank-r exact decomposition by starting from a random 
    subset of schoolbook columns and refining with ALS.
    """
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Initialize by selecting 'rank' random columns from schoolbook
        indices = np.random.choice(n * m * p, size=rank, replace=False)
        U = U_sb[:, indices].copy()
        V = V_sb[:, indices].copy()
        W = W_sb[:, indices].copy()
        
        # Add small noise to break symmetry and allow movement in null space
        U += np.random.randn(n * m, rank) * 1e-4
        V += np.random.randn(m * p, rank) * 1e-4
        W += np.random.randn(p * n, rank) * 1e-4
        
        # ALS Iterations
        prev_err = float('inf')
        for it in range(100):
            if time.time() > start_time + time_limit:
                break
                
            # Update U
            U_new = np.zeros_like(U)
            for idx in range(n * m):
                i, j = divmod(idx, m)
                V_part = V[j*p:(j+1)*p, :]
                W_part = W[np.arange(p)*n + i, :]
                A = V_part * W_part
                b = np.ones(p)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except:
                    U_new[idx, :] = U[idx, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for idx in range(m * p):
                j, k = divmod(idx, p)
                U_part = U[np.arange(n)*m + j, :]
                W_part = W[k*n:(k+1)*n, :]
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except:
                    V_new[idx, :] = V[idx, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for idx in range(p * n):
                k, i = divmod(idx, n)
                U_part = U[i*m:(i+1)*m, :]
                V_part = V[np.arange(m)*p + k, :]
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except:
                    W_new[idx, :] = W[idx, :]
            W = W_new
            
            # Compute error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-10:
                break
            if abs(prev_err - max_err) < 1e-12:
                break
            prev_err = max_err
        
        # Check final error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            # Try to snap to half-integers
            U_snapped, u_ok = _snap(U, 1e-4)
            V_snapped, v_ok = _snap(V, 1e-4)
            W_snapped, w_ok = _snap(W, 1e-4)
            
            if u_ok and v_ok and w_ok:
                if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
            
            # If snapping fails, try a polish step
            if max_err < 1e-6:
                U_polished, V_polished, W_polished = _polish(U, V, W, n, m, p, time_left=time_limit - (time.time() - start_time))
                if U_polished is not None:
                    return U_polished, V_polished, W_polished
                    
    return None

def _polish(U, V, W, n, m, p, time_left=5):
    """
    High-precision polish using Newton's method (Gauss-Seidel style).
    """
    if time_left < 0.5:
        return None, None, None
    
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(50):
        if time.time() - start > time_left:
            break
        
        # Gauss-Seidel updates
        for idx in range(n * m):
            i, j = divmod(idx, m)
            res = np.zeros(p)
            for k in range(p):
                val = np.sum(U[idx, :] * V[j*p+k, :] * W[k*n+i, :])
                res[k] = 1.0 - val
            
            V_part = V[j*p:(j+1)*p, :]
            W_part = W[np.arange(p)*n + i, :]
            A = V_part * W_part
            
            try:
                delta = np.linalg.solve(A, res)
                U[idx, :] += delta
            except:
                pass
        
        for idx in range(m * p):
            j, k = divmod(idx, p)
            res = np.zeros(n)
            for i in range(n):
                val = np.sum(U[i*m+j, :] * V[idx, :] * W[k*n+i, :])
                res[i] = 1.0 - val
            
            U_part = U[np.arange(n)*m + j, :]
            W_part = W[k*n:(k+1)*n, :]
            A = U_part * W_part
            
            try:
                delta = np.linalg.solve(A, res)
                V[idx, :] += delta
            except:
                pass
        
        for idx in range(p * n):
            k, i = divmod(idx, n)
            res = np.zeros(m)
            for j in range(m):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[idx, :])
                res[j] = 1.0 - val
            
            U_part = U[i*m:(i+1)*m, :]
            V_part = V[np.arange(m)*p + k, :]
            A = U_part * V_part
            
            try:
                delta = np.linalg.solve(A, res)
                W[idx, :] += delta
            except:
                pass
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-12:
            break
    
    # Final snap
    U_snapped, u_ok = _snap(U, 1e-3)
    V_snapped, v_ok = _snap(V, 1e-3)
    W_snapped, w_ok = _snap(W, 1e-3)
    
    if u_ok and v_ok and w_ok:
        if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
    
    return None, None, None

def _snap(arr, tol=1e-6):
    """Snap to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
