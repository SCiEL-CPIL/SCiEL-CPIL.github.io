import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _target_tensor(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _schoolbook(n, m, p):
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

def _to_list(M):
    return M.tolist()

def _find_rank_15_decomp(n2, m2, p2, deadline):
    """
    Finds a rank-15 decomposition for 2x3x3 multiplication tensor.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    R = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(rows_u, R) * 0.5
        V = np.random.randn(rows_v, R) * 0.5
        W = np.random.randn(rows_w, R) * 0.5
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Solve for U
            # T_flat_u shape: (rows_u, rows_v*rows_w)
            # Z_u shape: (R, rows_v*rows_w)
            # U @ Z_u.T = T_flat_u.T => Z_u @ U.T = T_flat_u
            Z_u = np.zeros((R, rows_v * rows_w))
            for r in range(R):
                Z_u[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat_u = T233.transpose(0, 1, 2).reshape(rows_u, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z_u.T, T_flat_u.T, rcond=None)
                U = U_new.T
            except:
                break

            # Solve for V
            Z_v = np.zeros((R, rows_u * rows_w))
            for r in range(R):
                Z_v[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            T_flat_v = T233.transpose(1, 0, 2).reshape(rows_v, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                break

            # Solve for W
            Z_w = np.zeros((R, rows_u * rows_v))
            for r in range(R):
                Z_w[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            T_flat_w = T233.transpose(2, 0, 1).reshape(rows_w, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                break
            
        # Snap to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T233, atol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140  # Safety margin
    
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}
    
    # Attempt rank 30 decomposition
    # 2x3x6 can be split into two 2x3x3 blocks
    n2, m2, p2 = 2, 3, 3
    decomp_233 = _find_rank_15_decomp(n2, m2, p2, deadline)
    
    if decomp_233 is not None:
        U233, V233, W233 = decomp_233
        R233 = 15
        R236 = 2 * R233
        
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1: k=0,1,2
        U236[:, :R233] = U233
        V236[:m2*p2, :R233] = V233
        W236[:p2*n2, :R233] = W233
        
        # Block 2: k=3,4,5
        # Note: The U matrix (input A) is the same.
        # V matrix (input B) takes the second half of columns.
        # W matrix (output C) takes the second half of columns.
        U236[:, R233:] = U233
        V236[m2*p2:, R233:] = V233
        W236[p2*n2:, R233:] = W233
        
        # Verify full tensor
        if _verify_decomposition(U236, V236, W236, n, m, p):
            return {
                "U": _to_list(U236),
                "V": _to_list(V236),
                "W": _to_list(W236)
            }
            
    return best_result
