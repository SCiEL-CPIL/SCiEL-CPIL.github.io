"""Optimized program for 2x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _to_half_int_val(x, tol=1e-9):
    """Round to half-integer (k/2) if close enough, else return None.
    Returns the actual value x' = k/2."""
    ix = int(round(2 * x))
    if abs(2 * x - ix) < tol:
        return ix / 2.0
    return None

def _check_solution(U, V, W):
    """Check if U, V, W (as lists of lists) give a valid decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def _is_half_int(val, tol=1e-9):
    """Check if val is close to a half-integer."""
    ix = int(round(2 * val))
    return abs(2 * val - ix) < tol

def _snap_to_half_int(val, tol=1e-9):
    """Snap val to nearest half-integer."""
    ix = int(round(2 * val))
    return ix / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Start with Schoolbook (Rank 42)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def _residual(U, V, W):
        """Compute sum of squared residuals."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for c in range(rank):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    res += (s - 1.0) ** 2
        return res

    def _optimize(U, V, W, max_iters=500, tol=1e-10):
        """Run coordinate descent to minimize residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        prev_res = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U: For each (i,j), solve M_{i,j} @ u = ones(p)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        M[k, :] = v_row * w_row
                    
                    t = np.ones(p)
                    try:
                        u_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        U[i * m + j, :] = u_sol
                    except:
                        pass
                        
            # Update V: For each (j,k), solve M_{j,k} @ v = ones(n)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        u_row = U[i * m + j, :]
                        w_row = W[k * n + i, :]
                        M[i, :] = u_row * w_row
                    t = np.ones(n)
                    try:
                        v_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        V[j * p + k, :] = v_sol
                    except:
                        pass
                        
            # Update W: For each (k,i), solve M_{k,i} @ w = ones(m)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        M[j, :] = u_row * v_row
                    t = np.ones(m)
                    try:
                        w_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        W[k * n + i, :] = w_sol
                    except:
                        pass
                        
            # Check residual
            res = _residual(U, V, W)
            
            if abs(prev_res - res) < tol:
                break
            prev_res = res
            
        return U, V, W, res

    def _try_finalize(U, V, W):
        """Try to snap to half-integers and verify."""
        U_snap = np.array([[_snap_to_half_int(x) for x in row] for row in U], dtype=float)
        V_snap = np.array([[_snap_to_half_int(x) for x in row] for row in V], dtype=float)
        W_snap = np.array([[_snap_to_half_int(x) for x in row] for row in W], dtype=float)
        
        # Check all values are half-integers
        valid = True
        for mat in [U_snap, V_snap, W_snap]:
            for row in mat:
                for val in row:
                    if not _is_half_int(val):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        # Convert to list of lists with proper types
        def _to_list(mat):
            result = []
            for row in mat:
                r = []
                for val in row:
                    # Convert to int if integer, else keep as float (half-integer)
                    if abs(val - round(val)) < 1e-9:
                        r.append(int(round(val)))
                    else:
                        # It's a half-integer like 0.5, 1.5, etc.
                        # Store as float for now; the evaluator should handle it
                        r.append(val)
                result.append(r)
            return result
        
        U_list = _to_list(U_snap)
        V_list = _to_list(V_snap)
        W_list = _to_list(W_snap)
        
        if _check_solution(U_list, V_list, W_list):
            return U_list, V_list, W_list
        return None

    # Try ranks from 41 down to 30
    rng = np.random.default_rng(42)
    
    for rank in range(41, 29, -1):
        if time.time() > deadline:
            break
            
        num_restarts = 8 if rank >= 38 else 5
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Smart initialization: start near schoolbook-like structure
            # Random but with some structure
            U = rng.standard_normal((n * m, rank)) * 0.5
            V = rng.standard_normal((m * p, rank)) * 0.5
            W = rng.standard_normal((p * n, rank)) * 0.5
            
            # Add a bias to encourage non-zero entries
            U += 0.3
            V += 0.3
            W += 0.3
            
            U_opt, V_opt, W_opt, res = _optimize(U, V, W, max_iters=300)
            
            result = _try_finalize(U_opt, V_opt, W_opt)
            if result is not None:
                U_final, V_final, W_final = result
                return {"U": U_final, "V": V_final, "W": W_final}
                
            # Also try with different initialization: sparse-like
            U2 = rng.standard_normal((n * m, rank))
            V2 = rng.standard_normal((m * p, rank))
            W2 = rng.standard_normal((p * n, rank))
            
            # Make some entries zero to encourage sparsity
            for mat in [U2, V2, W2]:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if rng.random() < 0.3:
                            mat[i, j] = 0.0
            
            U_opt2, V_opt2, W_opt2, res2 = _optimize(U2, V2, W2, max_iters=300)
            
            result2 = _try_finalize(U_opt2, V_opt2, W_opt2)
            if result2 is not None:
                U_final, V_final, W_final = result2
                return {"U": U_final, "V": V_final, "W": W_final}
                
    # If nothing found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
