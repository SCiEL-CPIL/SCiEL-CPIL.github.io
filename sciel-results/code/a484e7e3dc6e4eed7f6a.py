import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
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
    
    # 2. Try Winograd-based decomposition for 3x4x4
    # For 2x2 blocks, Winograd reduces 4 mults to 3
    # We can apply this to overlapping 2x2 sub-blocks
    
    # Strategy: Use Winograd for the 2x2 core, then extend to 3x4x4
    # A 3x4 matrix times 4x4 matrix:
    # Row blocks of A: [0:2], [1:3] (2 blocks)
    # Col blocks of B: [0:2], [1:3], [2:4] (3 blocks)
    # But we need to be careful about overlaps
    
    # Let's try a structured approach:
    # For each pair of row-blocks and col-blocks, use Winograd
    # Row block 0: rows 0,1 of A
    # Row block 1: rows 1,2 of A
    # Col block 0: cols 0,1 of B
    # Col block 1: cols 1,2 of B  
    # Col block 2: cols 2,3 of B
    
    # But this double-counts some entries. Let's think differently.
    
    # Alternative: Just try to find rank < 48 using optimized search
    # Start from rank 47 down
    
    for rank in range(best_rank - 1, max(n, m, p) - 1, -1):
        if time.time() > deadline - 5:
            break
            
        found = _try_find_rank(n, m, p, rank, max_restarts=30, time_limit=8)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            break
            
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10):
    """
    Attempts to find a rank-r exact decomposition with half-integer entries.
    Uses optimized ALS with better initialization and convergence criteria.
    """
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Better initialization: use structured init based on schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with random small values plus some structure
        for r in range(rank):
            # Assign each tensor entry to a random factor
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            U[i*m+j, r] += random.choice([-1, 1])
                        if random.random() < 0.3:
                            V[j*p+k, r] += random.choice([-1, 1])
                        if random.random() < 0.3:
                            W[k*n+i, r] += random.choice([-1, 1])
        
        # Add small random noise to break symmetry
        U += np.random.randn(n * m, rank) * 0.1
        V += np.random.randn(m * p, rank) * 0.1
        W += np.random.randn(p * n, rank) * 0.1
        
        # ALS Iterations with early stopping
        prev_err = float('inf')
        for it in range(80):
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
            
            # Compute error and check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                break
            if abs(prev_err - max_err) < 1e-10:
                break
            prev_err = max_err
        
        # Check final error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Try to snap to half-integers
            U_snapped, u_ok = _snap(U, 1e-4)
            V_snapped, v_ok = _snap(V, 1e-4)
            W_snapped, w_ok = _snap(W, 1e-4)
            
            if u_ok and v_ok and w_ok:
                if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
            
            # If snapping fails, try a polish step with tighter tolerance
            if max_err < 1e-5:
                U_polished, V_polished, W_polished = _polish(U, V, W, n, m, p, time_left=time_limit - (time.time() - start_time))
                if U_polished is not None:
                    return U_polished, V_polished, W_polished
                    
    return None

def _polish(U, V, W, n, m, p, time_left=5):
    """
    High-precision polish: use Newton's method to drive error to machine precision,
    then snap to half-integers.
    """
    if time_left < 0.5:
        return None, None, None
    
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(20):
        if time.time() - start > time_left:
            break
        
        # Compute residual for each tensor entry
        # For entry (i,j,k): residual = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Gradient w.r.t. U[i*m+j,c]: -residual * V[j*p+k,c]*W[k*n+i,c]
        # This is a coupled system; use Gauss-Seidel style updates
        
        changed = False
        for idx in range(n * m):
            i, j = divmod(idx, m)
            # Collect residuals for this (i,j)
            res = np.zeros(p)
            for k in range(p):
                val = np.sum(U[idx, :] * V[j*p+k, :] * W[k*n+i, :])
                res[k] = 1.0 - val
            
            # Build A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            V_part = V[j*p:(j+1)*p, :]
            W_part = W[np.arange(p)*n + i, :]
            A = V_part * W_part
            
            # Solve A @ delta = res
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
        
        if max_err < 1e-10:
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
