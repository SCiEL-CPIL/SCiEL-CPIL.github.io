import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
SCHOOLBOOK_RANK = n * m * p

def _schoolbook():
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(W[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_mat = U[:, c].reshape((n, m))
        v_mat = V[:, c].reshape((m, p))
        w_mat = W[:, c].reshape((p, n))
        
        for i in range(n):
            for j in range(m):
                u_val = u_mat[i, j]
                if u_val != 0:
                    w_vec = w_mat[:, i]
                    T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
                    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _snap_and_verify(U, V, W, step=0.5):
    """Snap to nearest multiples of step and verify."""
    U_snap = np.round(U / step) * step
    V_snap = np.round(V / step) * step
    W_snap = np.round(W / step) * step
    
    if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    return None

def _try_rank_als(r, time_limit=5.0, seed=None):
    """
    Try to find a decomposition of rank r using alternating least squares.
    Uses vectorized operations for speed.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random values
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        # Normalize columns
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            total_norm = norm_u * norm_v * norm_w
            if total_norm > 1e-9:
                scale = (1.0 / total_norm) ** (1.0/3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
        
        # ALS iterations
        best_residual = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(30):
            if time.time() - start_time > time_limit:
                break
                
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    try:
                        U_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m + j, :] = U_vec
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        V_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p + k, :] = V_vec
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n + i, :] = W_vec
                    except:
                        pass
            
            # Compute residual
            T = np.zeros((n, m, p))
            for c in range(r):
                u_mat = U[:, c].reshape((n, m))
                v_mat = V[:, c].reshape((m, p))
                w_mat = W[:, c].reshape((p, n))
                for i in range(n):
                    for j in range(m):
                        u_val = u_mat[i, j]
                        if u_val != 0:
                            w_vec = w_mat[:, i]
                            T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
            
            residual = np.sum((T - 1.0)**2)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-8:
                # Try snapping
                result = _snap_and_verify(U, V, W, step=0.5)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
                result = _snap_and_verify(U, V, W, step=1.0)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
        
        # After ALS, try snapping the best solution
        if best_U is not None:
            result = _snap_and_verify(best_U, best_V, best_W, step=0.5)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}
            result = _snap_and_verify(best_U, best_V, best_W, step=1.0)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}

    return None

def _try_rank_structured(r, time_limit=3.0, seed=None):
    """
    Try structured initializations that might lead to exact solutions.
    Uses patterns based on subsets of indices.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Try several structured patterns
    patterns = 0
    
    for pattern_idx in range(20):
        if time.time() - start_time > time_limit:
            break
        
        patterns += 1
        
        # Pattern 1: Random but with some sparsity
        if pattern_idx < 5:
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            W = rng.standard_normal((p * n, r))
            # Add sparsity
            for c in range(r):
                mask_u = rng.random(n * m) < 0.3
                mask_v = rng.random(m * p) < 0.3
                mask_w = rng.random(p * n) < 0.3
                U[mask_u, c] = 0
                V[mask_v, c] = 0
                W[mask_w, c] = 0
        else:
            # Pattern 2: Based on Kronecker products or other structures
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Try to assign each column to a specific (i,j,k) triple with variations
            for c in range(r):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                
                # Set primary entry
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                
                # Add small perturbations to nearby indices
                if c % 3 == 0 and i + 1 < n:
                    U[(i+1) * m + j, c] = rng.standard_normal() * 0.5
                if c % 3 == 1 and j + 1 < m:
                    V[j * p + k, c] += rng.standard_normal() * 0.5
                if c % 3 == 2 and k + 1 < p:
                    W[k * n + i, c] += rng.standard_normal() * 0.5
        
        # Normalize
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            total_norm = norm_u * norm_v * norm_w
            if total_norm > 1e-9:
                scale = (1.0 / total_norm) ** (1.0/3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
        
        # Quick ALS polish (fewer iterations)
        for it in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check residual
            T = np.zeros((n, m, p))
            for c in range(r):
                u_mat = U[:, c].reshape((n, m))
                v_mat = V[:, c].reshape((m, p))
                w_mat = W[:, c].reshape((p, n))
                for i in range(n):
                    for j in range(m):
                        u_val = u_mat[i, j]
                        if u_val != 0:
                            w_vec = w_mat[:, i]
                            T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
            
            residual = np.sum((T - 1.0)**2)
            if residual < 1e-8:
                result = _snap_and_verify(U, V, W, step=0.5)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
                result = _snap_and_verify(U, V, W, step=1.0)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}

    return None

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin for 150s limit
    
    # Search for ranks from 59 downwards
    # Allocate more time to promising ranks
    
    # First, try a quick check with structured approach for higher ranks
    # Then use ALS for lower ranks
    
    best_result = None
    
    # Try ranks from 59 down to 40 with structured approach first
    for r in range(59, 39, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Time budget: give more time to lower ranks
        time_budget = 3.0 if r >= 50 else 2.0
        
        # Try structured approach
        result = _try_rank_structured(r, time_limit=time_budget, seed=r)
        if result is not None:
            return result
        
        # Try ALS approach
        result = _try_rank_als(r, time_limit=time_budget, seed=r * 7)
        if result is not None:
            return result
    
    # If we have time, try even lower ranks
    if time.time() - start_time < 100:
        for r in range(39, 29, -1):
            if time.time() - start_time > DEADLINE:
                break
            
            time_budget = 2.0
            result = _try_rank_structured(r, time_limit=time_budget, seed=r + 100)
            if result is not None:
                return result
            
            result = _try_rank_als(r, time_limit=time_budget, seed=r * 13)
            if result is not None:
                return result
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start}s")
    print(f"Rank: {len(res['U'][0])}")
    if not _verify(res['U'], res['V'], res['W']):
        print("Verification Failed!")
    else:
        print("Verification Passed")
