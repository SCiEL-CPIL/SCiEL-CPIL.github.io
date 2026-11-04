"""
Program for 4x5 * 5x6 matrix multiplication tensor decomposition.
Uses a structured approach: start from schoolbook, then apply
greedy sparsification and local search to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 20, 30, 24


def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _snap_to_half(arr, tol=1e-6):
    """Snap entries to nearest half-integer if close enough."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return arr


def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    U = _snap_to_half(U, tol)
    V = _snap_to_half(V, tol)
    W = _snap_to_half(W, tol)
    return U, V, W


def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    return np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)


def _error(U, V, W):
    """Compute squared error of the decomposition."""
    T = _compute_tensor(U, V, W)
    return np.sum((T - 1.0) ** 2)


def _try_remove_column(U, V, W, col_idx, tol=1e-8):
    """
    Try to remove column col_idx by redistributing its contribution
    to other columns. Returns (new_U, new_V, new_W, success).
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, False
    
    # Get the contribution of the column to be removed
    u_col = U[:, col_idx].copy()
    v_col = V[:, col_idx].copy()
    w_col = W[:, col_idx].copy()
    
    # The tensor contribution from this column is:
    # T_col[i,j,k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # We need to find coefficients alpha_c for remaining columns c
    # such that sum_c alpha_c * u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] 
    # equals u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # This is a linear system. Let's set up the system.
    # For each (i,j,k) entry, we have:
    # sum_c alpha_c * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    
    # Number of equations: n*m*p = 120
    # Number of unknowns: rank - 1 (remaining columns)
    
    remaining_cols = [c for c in range(rank) if c != col_idx]
    n_unknowns = len(remaining_cols)
    n_equations = n * m * p
    
    if n_unknowns < n_equations:
        # Underdetermined or just determined - try to solve
        pass
    
    # Build the system A * alpha = b
    # A is n_equations x n_unknowns
    # b is n_equations
    
    A = np.zeros((n_equations, n_unknowns))
    b = np.zeros(n_equations)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for u_idx, c in enumerate(remaining_cols):
                    A[eq_idx, u_idx] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                b[eq_idx] = U[i * m + j, col_idx] * V[j * p + k, col_idx] * W[k * n + i, col_idx]
                eq_idx += 1
    
    # Solve the least squares problem
    try:
        alpha, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return U, V, W, False
    
    # Check if the solution is good
    residual = np.sum((A @ alpha - b) ** 2)
    if residual > 1e-10:
        return U, V, W, False
    
    # Check if alpha entries are close to half-integers
    alpha_snapped = _snap_to_half(alpha, tol=1e-4)
    if not np.allclose(alpha_snapped, alpha, atol=1e-4):
        # Not a clean solution, but still try
        alpha = alpha_snapped
    
    # Construct new matrices
    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    
    for u_idx, c in enumerate(remaining_cols):
        new_U[:, c] += alpha[u_idx] * u_col
        new_V[:, c] += alpha[u_idx] * v_col
        new_W[:, c] += alpha[u_idx] * w_col
    
    # Remove the column
    new_U = np.delete(new_U, col_idx, axis=1)
    new_V = np.delete(new_V, col_idx, axis=1)
    new_W = np.delete(new_W, col_idx, axis=1)
    
    return new_U, new_V, new_W, True


def _greedy_column_removal(U, V, W, deadline, max_removals=50):
    """
    Greedily try to remove columns one at a time.
    Returns the best (U, V, W) found.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for _ in range(max_removals):
        if time.time() > deadline - 5:
            break
        
        current_rank = best_U.shape[1]
        if current_rank <= max(nm, mp, pn) * 0.8:  # Don't go too low
            break
            
        best_success = False
        best_new_U, best_new_V, best_new_W = None, None, None
        best_col = -1
        
        # Try removing each column
        for col in range(current_rank):
            if time.time() > deadline - 5:
                break
                
            new_U, new_V, new_W, success = _try_remove_column(
                best_U, best_V, best_W, col
            )
            if success:
                # Verify the result
                if _verify(new_U, new_V, new_W, tol=1e-6):
                    best_success = True
                    best_new_U, best_new_V, best_new_W = new_U, new_V, new_W
                    best_col = col
                    break
        
        if best_success:
            best_U, best_V, best_W = best_new_U, best_new_V, best_new_W
            best_rank = best_U.shape[1]
        else:
            break
    
    return best_U, best_V, best_W


def _local_search(U, V, W, deadline, max_iter=5000, seed=0):
    """
    Local search to improve the decomposition.
    Tries to adjust entries to make the decomposition more sparse/clean.
    """
    rng = np.random.RandomState(seed)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(best_U, best_V, best_W)
    
    # Define allowed values
    vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    T_current = 0.1
    T_min = 1e-4
    alpha = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        if best_err < 1e-12:
            break
            
        # Pick a random matrix and entry
        mat_idx = rng.randint(0, 3)
        if mat_idx == 0:
            mat = best_U
            shape = best_U.shape
        elif mat_idx == 1:
            mat = best_V
            shape = best_V.shape
        else:
            mat = best_W
            shape = best_W.shape
            
        i = rng.randint(0, shape[0])
        j = rng.randint(0, shape[1])
        
        old_val = mat[i, j]
        new_val = rng.choice(vals)
        
        if new_val == old_val:
            continue
        
        mat[i, j] = new_val
        new_err = _error(best_U, best_V, best_W)
        
        if new_err < best_err:
            best_err = new_err
        elif T_current > T_min:
            delta = new_err - best_err
            if rng.rand() < np.exp(-delta / T_current):
                pass  # Accept
            else:
                mat[i, j] = old_val
                new_err = best_err
        else:
            mat[i, j] = old_val
            new_err = best_err
        
        T_current *= alpha
    
    return best_U, best_V, best_W, best_err


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Phase 1: Greedy column removal
    if time.time() < deadline - 30:
        U, V, W = _greedy_column_removal(U, V, W, deadline)
        if U.shape[1] < best_rank and _verify(U, V, W, tol=1e-6):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Phase 2: Local search to clean up
    if time.time() < deadline - 10:
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            U_try, V_try, W_try, err = _local_search(
                best_U, best_V, best_W, deadline, max_iter=3000, seed=seed
            )
            if err < 1e-10:
                U_snap, V_snap, W_snap = _to_integer_or_half(U_try, V_try, W_try)
                if _verify(U_snap, V_snap, W_snap, tol=1e-6):
                    if U_snap.shape[1] < best_rank:
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = U_snap.shape[1]
    
    # Phase 3: Try more aggressive column removal on the cleaned version
    if time.time() < deadline - 10:
        U2, V2, W2 = _greedy_column_removal(best_U, best_V, best_W, deadline, max_removals=20)
        if U2.shape[1] < best_rank and _verify(U2, V2, W2, tol=1e-6):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Final snap
    U_out, V_out, W_out = _to_integer_or_half(best_U, best_V, best_W)
    
    # Final verification
    if not _verify(U_out, V_out, W_out, tol=1e-6):
        # Fall back to schoolbook
        U_out, V_out, W_out = _schoolbook()
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
