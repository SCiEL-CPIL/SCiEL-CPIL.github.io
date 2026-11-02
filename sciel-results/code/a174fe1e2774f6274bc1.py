# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2.
    Returns U (4x7), V (4x7), W (4x7) with W indexed by [k*2+i, r]."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _residuals_small(U, V, W, n, m, p):
    """Compute residuals for a small tensor."""
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                idx += 1
    return res

def _find_rank10_223(deadline):
    """
    Try to find a rank-10 decomposition for 2x2x3.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Initial guess: Strassen (7) + Schoolbook (3) is rank 10?
    # Strassen covers 2 cols, Schoolbook covers 1 col.
    # Rank 7 + Rank 3 = 10.
    # But Strassen is for 2x2x2.
    # We can just use the known rank 11 (7+4) and try to optimize down to 10?
    # Or start from random?
    
    # Let's try to optimize from a rank 11 solution (Strassen + Schoolbook 1 col)
    # Strassen for k=0,1. Schoolbook for k=2.
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # First 7 cols: Strassen
    U11[:, :7] = U_str
    V11[:, :7] = V_str
    W11[:, :7] = W_str
    
    # Last 4 cols: Schoolbook for k=2
    # Wait, schoolbook for 2x2x1 has 4 terms.
    # We need to map them to the full 2x2x3 space.
    # U_sb is 4x4. V_sb is 2x4. W_sb is 2x4.
    # In the full space:
    # U indices: i*m+j -> 0..3
    # V indices: j*p+k -> j*3+2
    # W indices: k*n+i -> 2*2+i -> 4..5
    
    U11[:, 7:11] = U_sb
    # V_sb[j, t] corresponds to V[j*3+2, 7+t]
    for j in range(2):
        V11[j*3+2, 7:11] = V_sb[j, :]
    # W_sb[i, t] corresponds to W[2*2+i, 7+t]
    for i in range(2):
        W11[2*2+i, 7:11] = W_sb[i, :]
        
    if not _verify_decomposition(U11, V11, W11, n, m, p):
        # Fallback to random if construction fails
        U11 = np.random.randn(4, 11)
        V11 = np.random.randn(6, 11)
        W11 = np.random.randn(6, 11)
        
    def objective(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        res = _residuals_small(U, V, W, n, m, p)
        return np.sum(res**2)
        
    best_obj = np.inf
    best_params = None
    
    # Start from rank 11, zero out one column?
    # Or just random restarts.
    # Since 10 < 11, we are trying to "compress".
    
    # Try zeroing each of the last 4 columns of the rank-11 solution
    for zero_col in range(7, 11):
        if time.time() > deadline - 10:
            break
        x0 = np.concatenate([
            U11[:, :10].flatten(), # Drop column 10? No, we need to drop one specific column.
            # Actually, let's just take the first 10 columns of the 11-col solution?
            # That might not work well.
            # Better: take the 11 col solution, set one col to 0, and optimize.
            # But we have 10 cols in the target.
            # Let's just use random restarts.
        ])
        
    # Random restarts
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        # Scale random init
        U0 = np.random.randn(4, 10)
        V0 = np.random.randn(6, 10)
        W0 = np.random.randn(6, 10)
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is None or best_obj > 1e-8:
        return None, None, None
        
    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    res_snap = _residuals_small(U_snap, V_snap, W_snap, n, m, p)
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap
        
    # Check if raw solution is close to half-integer
    if np.allclose(U*2, np.round(U*2), atol=1e-5) and \
       np.allclose(V*2, np.round(V*2), atol=1e-5) and \
       np.allclose(W*2, np.round(W*2), atol=1e-5):
        return U_snap, V_snap, W_snap
        
    return None, None, None

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7

    # Last column: schoolbook for k=14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        U53, V53, W53 = _build_schoolbook(n, m, p)
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-10 decomposition for 2x2x3
    # If found, use it 5 times -> Rank 50
    U223, V223, W223 = _find_rank10_223(deadline - 20)
    if U223 is not None:
        # Construct 2x2x15 from 5 blocks of 2x2x3
        total_rank = 50
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for b in range(5):
            start_k = 3 * b
            offset = b * 10
            
            U[:, offset:offset+10] = U223
            for j in range(m):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V[j * p + k_global, offset:offset+10] = V223[j * 3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, offset:offset+10] = W223[k_local * 2 + i, :]
                    
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
# EVOLVE-BLOCK-END
