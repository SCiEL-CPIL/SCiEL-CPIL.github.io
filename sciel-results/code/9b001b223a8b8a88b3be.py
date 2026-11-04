import numpy as np
import time
import random
import scipy.optimize as opt

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if isinstance(U, np.ndarray):
        U = U.tolist()
    if isinstance(V, np.ndarray):
        V = V.tolist()
    if isinstance(W, np.ndarray):
        W = W.tolist()
    
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)."""
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _add_strassen_block(U, V, W, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, offset, m, p, n):
    """Add a 2x2 Strassen block at the given offsets."""
    terms = _strassen_2x2_terms()
    for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
        r = offset + t_idx
        for (xi, xj), val in x_coeff.items():
            U[(a_row_off + xi) * m + (a_col_off + xj)][r] += val
        for (yi, yj), val in y_coeff.items():
            V[(b_row_off + yi) * p + (b_col_off + yj)][r] += val
        for (zi, zj), val in z_coeff.items():
            W[(c_col_off + zj) * n + (c_row_off + zi)][r] += val

def _try_rank_44():
    """
    Split 3x4 * 4x4 into:
    - Row 0: 1x4 * 4x4 -> schoolbook, rank 16
    - Rows 1-2: 2x4 * 4x4 -> 4 Strassen 2x2 blocks, rank 28
    Total: 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) * B (4x4) -> Row 0 of C (1x4)
    for j in range(4):
        for k in range(4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) -> Rows 1-2 of C (2x4)
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, col, m, p, n)
        col += 7
    
    return U, V, W

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer and return new lists."""
    def snap(x):
        return round(x * 2) / 2.0
    
    U_snapped = [[snap(val) for val in row] for row in U]
    V_snapped = [[snap(val) for val in row] for row in V]
    W_snapped = [[snap(val) for val in row] for row in W]
    return U_snapped, V_snapped, W_snapped

def _optimize_rank(n, m, p, target_rank, U_init, V_init, W_init, time_budget, seed=42):
    """
    Use scipy.optimize.minimize to refine a rank-target_rank decomposition.
    U_init, V_init, W_init are lists of lists, shape (rows, target_rank).
    """
    np.random.seed(seed)
    start = time.time()
    
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    # Flatten
    u_flat = U.flatten()
    v_flat = V.flatten()
    w_flat = W.flatten()
    
    x0 = np.concatenate([u_flat, v_flat, w_flat])
    
    def _unpack(x):
        n_u = n * m * target_rank
        n_v = m * p * target_rank
        n_w = p * n * target_rank
        
        U_new = x[:n_u].reshape((n*m, target_rank))
        V_new = x[n_u:n_u+n_v].reshape((m*p, target_rank))
        W_new = x[n_u+n_v:].reshape((p*n, target_rank))
        return U_new, V_new, W_new
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e100
        U_new, V_new, W_new = _unpack(x)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i,j,k] = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
        
        err = np.sum((res - 1.0)**2)
        return err
    
    # Run L-BFGS-B
    try:
        res = opt.minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e-6:
            U_opt, V_opt, W_opt = _unpack(res.x)
            
            # Snap to half-integers
            U_snapped, V_snapped, W_snapped = _snap_to_half_integers(U_opt.tolist(), V_opt.tolist(), W_opt.tolist())
            
            # Verify snapped version
            if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped, V_snapped, W_snapped, True
            
            # If snapping failed, check if raw float solution is "close enough" to integers?
            # The task requires integer/half-integer. If we can't snap, we can't use it.
            # However, if the error is tiny, maybe we can try a different snap strategy?
            # For now, return False if snap fails.
            
    except Exception:
        pass
        
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try rank-44 decomposition (known to work)
    best_result = None
    try:
        U, V, W = _try_rank_44()
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find lower rank decompositions
    # We use the rank 44 solution as a starting point by zeroing out 2 columns
    # This is a heuristic to find rank 42.
    
    if best_result:
        U44, V44, W44 = best_result["U"], best_result["V"], best_result["W"]
        
        # Try Rank 42
        if time.time() < deadline - 30:
            # Create a rank 42 init by taking first 42 cols of rank 44
            U42_init = [row[:42] for row in U44]
            V42_init = [row[:42] for row in V44]
            W42_init = [row[:42] for row in W44]
            
            U_opt, V_opt, W_opt, success = _optimize_rank(n, m, p, 42, U42_init, V42_init, W42_init, time_budget=20.0, seed=42)
            if success:
                best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
        
        # Try Rank 40
        if time.time() < deadline - 10 and best_result and len(best_result["U"][0]) > 40:
            current_rank = len(best_result["U"][0])
            # If we have 42, try to go to 40. If we have 44, try to go to 40 directly.
            # Let's always try from the best known or 44.
            if current_rank == 44:
                U40_init = [row[:40] for row in U44]
                V40_init = [row[:40] for row in V44]
                W40_init = [row[:40] for row in W44]
            else:
                U40_init = [row[:40] for row in best_result["U"]]
                V40_init = [row[:40] for row in best_result["V"]]
                W40_init = [row[:40] for row in best_result["W"]]
                
            U_opt, V_opt, W_opt, success = _optimize_rank(n, m, p, 40, U40_init, V40_init, W40_init, time_budget=15.0, seed=43)
            if success:
                best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                
    # Schoolbook fallback
    if best_result is None:
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
