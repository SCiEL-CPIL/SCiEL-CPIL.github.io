"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

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

def _residuals_223(U, V, W):
    """Residuals for (2,2,3) tensor: 12 equations."""
    n, m, p = 2, 2, 3
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                idx += 1
    return res

def _get_m223_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for M_{2,2,3}.
    Strategy: start from rank-11 (Strassen + schoolbook), then optimize
    to satisfy the system with one fewer term by merging.
    Returns (U, V, W) with half-integer entries or None.
    """
    n, m, p = 2, 2, 3
    rank = 10

    # Build rank-11 starting point
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    U11[:, 0:7] = U_str
    for j in range(2):
        for k in range(2):
            V11[j*3+k, 0:7] = V_str[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W11[k*2+i, 0:7] = W_str[k*2+i, :]
    U11[:, 7:11] = U_sb
    for j in range(2):
        V11[j*3+2, 7:11] = V_sb[j, :]
    for i in range(2):
        W11[2*2+i, 7:11] = W_sb[i, :]

    # Verify rank-11 is correct
    r11 = _residuals_223(U11, V11, W11)
    if np.max(np.abs(r11)) > 1e-10:
        return None

    # Now try to find a rank-10 solution by optimization.
    # We'll parameterize U, V, W directly (160 params) and minimize sum of squares.
    # Use the rank-11 solution projected as an initial guess by zeroing one column
    # and adjusting, plus random perturbations.

    best_params = None
    best_obj = np.inf

    def objective(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        res = _residuals_223(U, V, W)
        return np.sum(res**2)

    # Seed 0: rank-11 with last column zeroed and spread
    x0_list = []
    x0_0 = np.zeros(160)
    # Copy rank-11 first 10 columns as initial guess
    x0_0[:40] = U11[:, 0:10].flatten()
    x0_0[40:100] = V11[:, 0:10].flatten()
    x0_0[100:160] = W11[:, 0:10].flatten()
    x0_list.append(x0_0)

    # Random seeds
    for seed in range(1, 12):
        np.random.seed(seed)
        x0_list.append(np.random.randn(160) * 0.8)

    for x0 in x0_list:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-12:
                break
        except Exception:
            continue

    if best_params is None or best_obj > 1e-8:
        return None

    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)

    # Snap to half-integers and re-verify
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    res_snap = _residuals_223(U_snap, V_snap, W_snap)
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap

    # If snapping fails, try the un-snapped if it's already integer/half-integer
    res_raw = _residuals_223(U, V, W)
    if np.max(np.abs(res_raw)) < 1e-6:
        # Check if entries are close to half-integers
        if np.allclose(U * 2, np.round(U * 2), atol=1e-6) and \
           np.allclose(V * 2, np.round(V * 2), atol=1e-6) and \
           np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return U_snap, V_snap, W_snap

    return None

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

def _build_rank50_from_223(U223, V223, W223):
    """Build full (2,2,15) rank-50 decomposition from (2,2,3) rank-10 blocks."""
    n, m, p = DIMENSIONS
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for b in range(5):
        start_k = 3 * b
        col_offset = 10 * b
        U[:, col_offset:col_offset+10] = U223
        for j in range(m):
            for k_local in range(3):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V223[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W223[k_local * 2 + i, :]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U53, V53, W53 = _build_schoolbook(n, m, p)
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-52 solution by optimizing from rank-53 warm start
    # Strategy: zero out one column of the rank-53 solution and optimize
    # the remaining 52 columns to satisfy the tensor equations.
    if time.time() < deadline - 5:
        U52, V52, W52 = _try_rank52(U53, V53, W53, deadline)
        if U52 is not None:
            if _verify_decomposition(U52, V52, W52, n, m, p, tol=1e-6):
                return {"U": U52.tolist(), "V": V52.tolist(), "W": W52.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

def _try_rank52(U53, V53, W53, deadline):
    """Try to find a rank-52 decomposition by optimizing from rank-53 warm start."""
    n, m, p = DIMENSIONS
    rank = 52
    
    def _residuals_full(U, V, W):
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                    idx += 1
        return res
    
    def objective(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        res = _residuals_full(U, V, W)
        return np.sum(res**2)
    
    # Warm start: use first 52 columns of rank-53 solution
    x0 = np.concatenate([
        U53[:, :rank].flatten(),
        V53[:, :rank].flatten(),
        W53[:, :rank].flatten()
    ])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < 1e-10:
            U = res.x[:n*m*rank].reshape(n*m, rank)
            V = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if np.max(np.abs(_residuals_full(U_snap, V_snap, W_snap))) < 1e-6:
                return U_snap, V_snap, W_snap
            # Check if raw solution is already half-integer
            if np.allclose(U*2, np.round(U*2), atol=1e-6) and \
               np.allclose(V*2, np.round(V*2), atol=1e-6) and \
               np.allclose(W*2, np.round(W*2), atol=1e-6):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
    
    return None, None, None
# EVOLVE-BLOCK-END
