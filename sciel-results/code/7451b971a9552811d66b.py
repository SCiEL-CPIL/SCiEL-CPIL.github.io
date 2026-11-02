import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _als_step_U(U, V, W, n, m, p, R):
    """
    Update U using least squares.
    U shape: (n*m, R)
    V shape: (m*p, R)
    W shape: (p*n, R)
    """
    new_U = np.zeros_like(U)
    # For each (i, j), we solve for U[i*m+j, :]
    # The equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a linear system for U[i*m+j, :] for each k?
    # No, the sum is over r. The equations are for each k.
    # So for fixed i, j, we have p equations for R unknowns U[i*m+j, r].
    # A * x = b, where A is p x R, x is R x 1, b is p x 1 (all ones).
    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            A = np.zeros((p, R))
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                A[k, :] = V[idx_v, :] * W[idx_w, :]
            b = np.ones(p)
            # Solve A x = b
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_U[idx_u, :] = x
    return new_U

def _als_step_V(U, V, W, n, m, p, R):
    """
    Update V using least squares.
    V shape: (m*p, R)
    """
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
    """
    Update W using least squares.
    W shape: (p*n, R)
    """
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

def _calculate_error(U, V, W, n, m, p, R):
    """
    Calculate the Frobenius norm of the error.
    """
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
    """
    Snap values to the nearest half-integer.
    """
    snapped = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            # Round to nearest 0.5
            snapped[i, j] = round(val * 2) / 2.0
    return snapped

def _verify_exact(U, V, W, n, m, p, R):
    """
    Verify the decomposition exactly using Fractions.
    """
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

def _find_decomposition_sub(n, m, p, R, max_time=30):
    """
    Find a rank-R decomposition for M_{n,m,p} using ALS.
    Returns (U, V, W) as lists of lists of floats (snapped) or None.
    """
    start_time = time.time()
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() - start_time > max_time:
            break
            
        np.random.seed(42 + trial)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to avoid large numbers initially?
        # The values should be around 1/sqrt(R) maybe?
        # Let's just run ALS.
        
        for it in range(100):
            if time.time() - start_time > max_time:
                break
                
            U = _als_step_U(U, V, W, n, m, p, R)
            V = _als_step_V(U, V, W, n, m, p, R)
            W = _als_step_W(U, V, W, n, m, p, R)
            
            if it % 10 == 0:
                err = _calculate_error(U, V, W, n, m, p, R)
                if err < 1e-8:
                    # Try to snap
                    U_snap = _snap_to_half_integers(U)
                    V_snap = _snap_to_half_integers(V)
                    W_snap = _snap_to_half_integers(W)
                    
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, R):
                        # Convert to lists of lists
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Target: Rank 28 for M_{2,3,6}
    # Strategy: Find Rank 14 for M_{2,3,3} and tile it.
    
    # Sub-problem dimensions
    n_sub, m_sub, p_sub = 2, 3, 3
    R_sub = 14
    
    # Try to find the sub-decomposition
    result_sub = _find_decomposition_sub(n_sub, m_sub, p_sub, R_sub, max_time=100)
    
    if result_sub is not None:
        U_sub, V_sub, W_sub = result_sub
        # Construct full decomposition
        R_full = R_sub * 2
        
        U_full = [[0] * R_full for _ in range(n * m)]
        V_full = [[0] * R_full for _ in range(m * p)]
        W_full = [[0] * R_full for _ in range(p * n)]
        
        # Fill U
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                for r in range(R_sub):
                    U_full[idx_u][r] = U_sub[idx_u][r]
                    U_full[idx_u][r + R_sub] = U_sub[idx_u][r]
        
        # Fill V
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                # Map k to k_sub
                if k < p_sub:
                    k_sub = k
                    r_offset = 0
                else:
                    k_sub = k - p_sub
                    r_offset = R_sub
                
                idx_v_sub = j * p_sub + k_sub
                for r in range(R_sub):
                    V_full[idx_v][r + r_offset] = V_sub[idx_v_sub][r]
        
        # Fill W
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                # Map k to k_sub
                if k < p_sub:
                    k_sub = k
                    r_offset = 0
                else:
                    k_sub = k - p_sub
                    r_offset = R_sub
                    
                idx_w_sub = k_sub * n_sub + i
                for r in range(R_sub):
                    W_full[idx_w][r + r_offset] = W_sub[idx_w_sub][r]
        
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
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
