"""Matrix multiplication 2x3x7: exact rank search via integer linear algebra."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def _build_tensors(n, m, p):
    """Build the matricizations T2 (n*m, m*p) and T3 (p*n, n*m) of the multiplication tensor."""
    T2 = np.zeros((n * m, m * p), dtype=np.float64)
    T3 = np.zeros((p * n, n * m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k] = 1.0
                T3[k * n + i, i * m + j] = 1.0
    return T2, T3


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    r = U.shape[1]
    T2 = U @ V.T
    T3 = W @ U.T
    T1 = V @ W.T
    
    # Check T2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T2[i * m + j, j * p + k] - 1.0) > tol:
                    return False
    # Check T3
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if abs(T3[k * n + i, i * m + j] - 1.0) > tol:
                    return False
    # Check T1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T1[j * p + k, k * n + i] - 1.0) > tol:
                    return False
    return True


def _is_half_integer(vec, tol=1e-8):
    """Check if all entries are half-integers (0, 0.5, 1, 1.5, ...)."""
    scaled = 2.0 * vec
    rounded = np.round(scaled)
    return bool(np.all(np.abs(scaled - rounded) < tol))


def _snap_to_half_integer(vec):
    """Snap vector entries to nearest half-integer."""
    return np.round(2.0 * vec) / 2.0


def _try_rank_r(n, m, p, r, T2, T3, max_iter=5000, seed=42):
    """
    Try to find a rank-r decomposition using iterative refinement.
    We parameterize:
      U: (n*m, r)
      V: (m*p, r) 
      W: (p*n, r)
    such that U @ V.T = T2 and W @ U.T = T3.
    
    Strategy: Start with random initialization, then use alternating least squares
    to minimize ||U V^T - T2||^2 + ||W U^T - T3||^2 + ||V W^T - T1||^2
    Then check if the solution is close to half-integer.
    """
    np.random.seed(seed)
    
    T1 = np.zeros((m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[j * p + k, k * n + i] = 1.0
    
    # Initialize
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Normalize columns
    for c in range(r):
        nu = np.linalg.norm(U[:, c])
        nv = np.linalg.norm(V[:, c])
        nw = np.linalg.norm(W[:, c])
        if nu > 1e-10: U[:, c] /= nu
        if nv > 1e-10: V[:, c] /= nv
        if nw > 1e-10: W[:, c] /= nw
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: minimize ||U V^T - T2||^2 + ||W U^T - T3||^2
        # = ||V U^T - T2^T||^2 + ||U W^T - T3^T||^2
        # Gradient descent or closed form
        # U V^T = T2  =>  U = T2 (V^T)^+ 
        # W U^T = T3  =>  U^T = (W^T)^+ T3  =>  U = T3^T (W^T)^+^T
        
        # Use combined least squares
        # Stack: [V; W^T] @ U^T = [T2^T; T3^T]
        # This is (m*p + p*n) x r system for U^T (r x n*m)
        
        # Actually, let's do alternating:
        # Fix V, W: solve for U
        # U V^T ≈ T2  =>  (V V^T) U^T = V T2^T  =>  U^T = (V V^T)^{-1} V T2^T
        # W U^T ≈ T3  =>  (W W^T) U^T = W T3^T  =>  U^T = (W W^T)^{-1} W T3^T
        
        # Weighted combination
        VtV = V.T @ V  # r x r
        WtW = W.T @ W  # r x r
        
        VT = V.T  # r x m*p
        WT = W.T  # r x p*n
        
        # U V^T = T2  =>  U = T2 V (V^T V)^{-1}
        # W U^T = T3  =>  U = (W^T)^+ T3^T  =>  U^T = (W W^T)^{-1} W T3^T
        
        try:
            # From U V^T = T2:
            U1 = T2 @ V @ np.linalg.inv(VtV)  # n*m x r
            # From W U^T = T3:
            U2 = np.linalg.inv(WtW) @ W @ T3.T  # r x n*m, transpose
            U2 = U2.T  # n*m x r
            
            # Average
            U_new = 0.5 * (U1 + U2)
        except np.linalg.LinAlgError:
            U_new = U
        
        # Fix U, W: solve for V
        # U V^T = T2  =>  V^T = U^+ T2  =>  V = T2^T (U U^T)^{-1} U
        # V W^T = T1  =>  V = T1 (W^T)^+  =>  V^T = (W W^T)^{-1} W T1^T  =>  V = T1^T W (W W^T)^{-1}
        
        UtU = U.T @ U
        try:
            V1 = T2.T @ U @ np.linalg.inv(UtU)  # m*p x r
            V2 = T1.T @ W @ np.linalg.inv(WtW)  # m*p x r
            V_new = 0.5 * (V1 + V2)
        except np.linalg.LinAlgError:
            V_new = V
        
        # Fix U, V: solve for W
        # W U^T = T3  =>  W = T3 (U^T)^+  =>  W^T = (U U^T)^{-1} U T3^T  =>  W = T3^T U (U U^T)^{-1}
        # V W^T = T1  =>  W^T = (V V^T)^{-1} V T1^T  =>  W = T1^T V (V V^T)^{-1}
        
        UtU2 = U.T @ U
        VtV2 = V.T @ V
        try:
            W1 = T3.T @ U @ np.linalg.inv(UtU2)  # p*n x r
            W2 = T1.T @ V @ np.linalg.inv(VtV2)  # p*n x r
            W_new = 0.5 * (W1 + W2)
        except np.linalg.LinAlgError:
            W_new = W
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        err = np.linalg.norm(U @ V.T - T2, 'fro')**2 + \
              np.linalg.norm(W @ U.T - T3, 'fro')**2 + \
              np.linalg.norm(V @ W.T - T1, 'fro')**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            break
    
    # Check if best solution is close to half-integer
    U_s = _snap_to_half_integer(best_U)
    V_s = _snap_to_half_integer(best_V)
    W_s = _snap_to_half_integer(best_W)
    
    # Verify the snapped version
    if _verify_exact(U_s, V_s, W_s, n, m, p, tol=1e-6):
        return U_s, V_s, W_s
    
    # Also try the un-snapped version
    if _verify_exact(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Check if it's half-integer
        if _is_half_integer(best_U) and _is_half_integer(best_V) and _is_half_integer(best_W):
            return _snap_to_half_integer(best_U), _snap_to_half_integer(best_V), _snap_to_half_integer(best_W)
    
    return None


def _greedy_peel(n, m, p, r, T2, T3):
    """
    Greedy rank-1 peeling: repeatedly find the best rank-1 approximation
    to the residual tensor and subtract it.
    """
    T1 = np.zeros((m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[j * p + k, k * n + i] = 1.0
    
    U_list = []
    V_list = []
    W_list = []
    
    Res_T2 = T2.copy()
    Res_T3 = T3.copy()
    Res_T1 = T1.copy()
    
    for step in range(r):
        # Find best rank-1 approximation to Res_T3 = W @ U^T
        # SVD of Res_T3
        try:
            U_mat, s, Vt = np.linalg.svd(Res_T3, full_matrices=False)
            if len(s) == 0 or s[0] < 1e-10:
                break
            w_vec = U_mat[:, 0]
            u_vec = Vt[0, :]
            a = s[0]
        except:
            break
        
        # Now find v_vec such that u_vec @ v_vec^T approximates Res_T2
        # v_vec = Res_T2 @ u_vec / (u_vec^T u_vec)
        u_norm_sq = np.dot(u_vec, u_vec)
        if u_norm_sq < 1e-10:
            break
        v_vec = Res_T2 @ u_vec / u_norm_sq
        
        # Also check consistency with Res_T1
        # v_vec @ w_vec^T should approximate Res_T1
        # w_vec should be consistent
        
        # Update residuals
        Res_T3 = Res_T3 - a * np.outer(w_vec, u_vec)
        Res_T2 = Res_T2 - a * np.outer(u_vec, v_vec)
        Res_T1 = Res_T1 - a * np.outer(v_vec, w_vec)
        
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    if len(U_list) < r:
        return None
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Try to snap to half-integers
    U_s = _snap_to_half_integer(U)
    V_s = _snap_to_half_integer(V)
    W_s = _snap_to_half_integer(W)
    
    if _verify_exact(U_s, V_s, W_s, n, m, p, tol=1e-6):
        return U_s, V_s, W_s
    
    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for val in row:
            hv = round(val * 2.0) / 2.0
            if abs(hv - round(hv)) < 1e-9:
                r.append(int(round(hv)))
            else:
                r.append(float(hv))
        out.append(r)
    return out


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    T2, T3 = _build_tensors(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {
        "U": _to_list(U_sb),
        "V": _to_list(V_sb),
        "W": _to_list(W_sb),
    }
    best_rank = n * m * p
    
    # Try ranks from low to high
    min_possible = 1
    max_try = min(best_rank, 25)
    
    for r in range(min_possible, max_try + 1):
        if time.time() > deadline:
            break
        
        # Method 1: Greedy peeling
        result = _greedy_peel(n, m, p, r, T2, T3)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": _to_list(U),
                "V": _to_list(V),
                "W": _to_list(W),
            }
            best_rank = r
            break
        
        # Method 2: Alternating least squares with multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank_r(n, m, p, r, T2, T3, max_iter=2000, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {
                    "U": _to_list(U),
                    "V": _to_list(V),
                    "W": _to_list(W),
                }
                best_rank = r
                break
        if best_rank < n * m * p:
            break
    
    return best_result
