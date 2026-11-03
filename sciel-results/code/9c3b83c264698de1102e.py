import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _als_step_U(U, V, W, n, m, p, R):
    """Update U using least squares."""
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            A = np.zeros((p, R))
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                A[k, :] = V[idx_v, :] * W[idx_w, :]
            b = np.ones(p)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_U[idx_u, :] = x
    return new_U

def _als_step_V(U, V, W, n, m, p, R):
    """Update V using least squares."""
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            A = np.zeros((n, R))
            for i in range(n):
                idx_u = i * m + j
                idx_w = k * n + i
                A[i, :] = U[idx_u, :] * W[idx_w, :]
            b = np.ones(n)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_V[idx_v, :] = x
    return new_V

def _als_step_W(U, V, W, n, m, p, R):
    """Update W using least squares."""
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            A = np.zeros((m, R))
            for j in range(m):
                idx_u = i * m + j
                idx_v = j * p + k
                A[j, :] = U[idx_u, :] * V[idx_v, :]
            b = np.ones(m)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_W[idx_w, :] = x
    return new_W

def _polish_to_exact(U, V, W, n, m, p, R, max_iters=50, tol=1e-12):
    """
    High-precision polish: iteratively refine entries to converge to exact solution.
    Uses tighter tolerance and more iterations for final convergence.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for it in range(max_iters):
        # Update U with tight tolerance
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                A = np.zeros((p, R))
                for k in range(p):
                    idx_v = j * p + k
                    idx_w = k * n + i
                    A[k, :] = V[idx_v, :] * W[idx_w, :]
                b = np.ones(p)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_U[idx_u, :] = x
        U = new_U
        
        # Update V
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                A = np.zeros((n, R))
                for i in range(n):
                    idx_u = i * m + j
                    idx_w = k * n + i
                    A[i, :] = U[idx_u, :] * W[idx_w, :]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_V[idx_v, :] = x
        V = new_V
        
        # Update W
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                A = np.zeros((m, R))
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    A[j, :] = U[idx_u, :] * V[idx_v, :]
                b = np.ones(m)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_W[idx_w, :] = x
        W = new_W
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    err += (val - 1.0) ** 2
        if np.sqrt(err) < tol:
            break
    
    return U, V, W

def _calculate_error(U, V, W, n, m, p, R):
    """Calculate the Frobenius norm of the error."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                err += (val - 1.0) ** 2
    return np.sqrt(err)

def _snap_to_half_integers(mat):
    """Snap values to the nearest half-integer."""
    snapped = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            snapped[i, j] = round(val * 2) / 2.0
    return snapped

def _verify_exact(U, V, W, n, m, p, R):
    """Verify the decomposition exactly using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    u_val = Fraction(str(U[idx_u, r]))
                    v_val = Fraction(str(V[idx_v, r]))
                    w_val = Fraction(str(W[idx_w, r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _find_2x3x3_rank11(max_time=60):
    """
    Find a rank-11 decomposition for 2x3x3 using ALS with exact verification.
    """
    n, m, p = 2, 3, 3
    R = 11
    start = time.time()
    
    for trial in range(100):
        if time.time() - start > max_time:
            break
        
        np.random.seed(1000 + trial)
        
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        for it in range(500):
            if time.time() - start > max_time:
                break
            
            U = _als_step_U(U, V, W, n, m, p, R)
            V = _als_step_V(U, V, W, n, m, p, R)
            W = _als_step_W(U, V, W, n, m, p, R)
            
            if it % 10 == 0:
                err = _calculate_error(U, V, W, n, m, p, R)
                if err < 1e-8:
                    # Polish to high precision before snapping
                    U_polished, V_polished, W_polished = _polish_to_exact(
                        U, V, W, n, m, p, R, max_iters=30, tol=1e-14)
                    
                    # Try to snap to half-integers
                    U_snap = _snap_to_half_integers(U_polished)
                    V_snap = _snap_to_half_integers(V_polished)
                    W_snap = _snap_to_half_integers(W_polished)
                    
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, R):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    
                    # Also try snapping original
                    U_snap2 = _snap_to_half_integers(U)
                    V_snap2 = _snap_to_half_integers(V)
                    W_snap2 = _snap_to_half_integers(W)
                    
                    if _verify_exact(U_snap2, V_snap2, W_snap2, n, m, p, R):
                        return U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()
    
    return None

def _tile_decomposition(U_sub, V_sub, W_sub, n, m, p, n_sub, m_sub, p_sub, R_sub):
    """
    Tile a decomposition of M_{n_sub, m_sub, p_sub} to M_{n, m, p}
    where n=n_sub, m=m_sub, and p is a multiple of p_sub.
    """
    repeats = p // p_sub
    R_full = R_sub * repeats
    
    U_full = [[0] * R_full for _ in range(n * m)]
    V_full = [[0] * R_full for _ in range(m * p)]
    W_full = [[0] * R_full for _ in range(p * n)]
    
    # Fill U: same pattern repeated for each tile
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for rep in range(repeats):
                r_offset = rep * R_sub
                for r in range(R_sub):
                    U_full[idx_u][r_offset + r] = U_sub[i * m_sub + j][r]
    
    # Fill V: each k-block uses the corresponding sub-V
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            rep = k // p_sub
            k_sub = k % p_sub
            r_offset = rep * R_sub
            idx_v_sub = j * p_sub + k_sub
            for r in range(R_sub):
                V_full[idx_v][r_offset + r] = V_sub[idx_v_sub][r]
    
    # Fill W: each k-block uses the corresponding sub-W
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            rep = k // p_sub
            k_sub = k % p_sub
            r_offset = rep * R_sub
            idx_w_sub = k_sub * n_sub + i
            for r in range(R_sub):
                W_full[idx_w][r_offset + r] = W_sub[idx_w_sub][r]
    
    return U_full, V_full, W_full

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy 1: Find rank-11 decomposition for 2x3x3, then tile to 2x3x6
    # This gives rank 22 (11 * 2)
    
    result_2x3x3 = _find_2x3x3_rank11(max_time=100)
    
    if result_2x3x3 is not None:
        U_sub, V_sub, W_sub = result_2x3x3
        n_sub, m_sub, p_sub = 2, 3, 3
        R_sub = 11
        
        U_full, V_full, W_full = _tile_decomposition(
            U_sub, V_sub, W_sub, n, m, p, n_sub, m_sub, p_sub, R_sub)
        
        # Verify the full decomposition
        U_np = np.array(U_full, dtype=float)
        V_np = np.array(V_full, dtype=float)
        W_np = np.array(W_full, dtype=float)
        R_full = R_sub * (p // p_sub)
        
        if _verify_exact(U_np, V_np, W_np, n, m, p, R_full):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
