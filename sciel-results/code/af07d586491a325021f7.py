import time
import numpy as np
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W give a valid tensor decomposition of matrix multiplication."""
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, r) or Va.shape != (m*p, r) or Wa.shape != (p*n, r):
        return False
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _residuals_3x3x3(params, r):
    """Residuals for 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    res = np.empty(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1.0
                idx += 1
    return res

def _jacobian_3x3x3(params, r):
    """Jacobian for 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r :].reshape(p*n, r)
    nres = n*m*p
    nparams = (n*m + m*p + p*n)*r
    J = np.zeros((nres, nparams))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    J[idx, i*m*r + j*r + c] = V[j*p+k, c] * W[k*n+i, c]
                for c in range(r):
                    J[idx, n*m*r + j*p*r + k*r + c] = U[i*m+j, c] * W[k*n+i, c]
                for c in range(r):
                    J[idx, n*m*r + m*p*r + k*n*r + i*r + c] = U[i*m+j, c] * V[j*p+k, c]
                idx += 1
    return J

def _find_3x3x3_decomposition(target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition for 3x3x3.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 3, 3, 3
    rng = np.random.RandomState(42)
    best_cost = np.inf
    best_params = None
    
    for restart in range(20):
        if time.time() > deadline:
            break
        U0 = rng.randn(n*m, target_rank)
        V0 = rng.randn(m*p, target_rank)
        W0 = rng.randn(p*n, target_rank)
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = least_squares(_residuals_3x3x3, x0, args=(target_rank,),
                                jac=_jacobian_3x3x3, method='lm', max_nfev=2000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_params = res.x
            if best_cost < 1e-10:
                break
        except Exception:
            continue

    if best_params is None or best_cost > 1e-8:
        return None

    U = best_params[:n*m*target_rank].reshape(n*m, target_rank)
    V = best_params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
    W = best_params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
    
    # Round to nearest 0.5
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), 3, 3, 3, tol=1e-6):
        return U, V, W
    return None

def _build_3x4x5_from_3x3x3(U3, V3, W3, U_sb_3x1x5, V_sb_3x1x5, W_sb_3x1x5, U_sb_3x3x2, V_sb_3x3x2, W_sb_3x3x2):
    """
    Combine 3x3x3 (rank r3), 3x1x5 (rank 15), 3x3x2 (rank 18) into 3x4x5.
    """
    r3 = U3.shape[1]
    r15 = U_sb_3x1x5.shape[1]
    r18 = U_sb_3x3x2.shape[1]
    total_rank = r3 + r15 + r18
    
    n, m, p = 3, 4, 5
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # 1. 3x3x3 part (rows 0-2 of A, cols 0-2 of B, rows 0-2 of C)
    # A is 3x4. We use A[:, 0:3].
    # B is 4x5. We use B[0:3, 0:3].
    # C is 3x5. We use C[0:3, 0:3].
    
    # U3 is (9, r3). Maps to A[i, j] for i,j in 0..2.
    # In full U, A[i, j] is at index i*4 + j.
    for i in range(3):
        for j in range(3):
            U_full[i*4 + j, 0:r3] = U3[i*3 + j, :]
            
    # V3 is (9, r3). Maps to B[j, k] for j,k in 0..2.
    # In full V, B[j, k] is at index j*5 + k.
    for j in range(3):
        for k in range(3):
            V_full[j*5 + k, 0:r3] = V3[j*3 + k, :]
            
    # W3 is (9, r3). Maps to C[k, i] for k,i in 0..2.
    # In full W, C[k, i] is at index k*3 + i.
    for k in range(3):
        for i in range(3):
            W_full[k*3 + i, 0:r3] = W3[k*3 + i, :]

    # 2. 3x1x5 part (rows 0-2 of A, col 3 of B, rows 0-2 of C, cols 0-4)
    # A[:, 3] * B[3, :] -> C[:, :]
    # U_sb_3x1x5 is (3, 15). Maps to A[i, 3].
    # In full U, A[i, 3] is at index i*4 + 3.
    for i in range(3):
        U_full[i*4 + 3, r3:r3+r15] = U_sb_3x1x5[i, :]
        
    # V_sb_3x1x5 is (1, 15). Maps to B[3, k].
    # In full V, B[3, k] is at index 3*5 + k = 15 + k.
    for k in range(5):
        V_full[15 + k, r3:r3+r15] = V_sb_3x1x5[0, :]
        
    # W_sb_3x1x5 is (5, 15). Maps to C[k, i].
    # In full W, C[k, i] is at index k*3 + i.
    W_full[0:15, r3:r3+r15] = W_sb_3x1x5

    # 3. 3x3x2 part (rows 0-2 of A, cols 0-2 of B, rows 0-2 of C, cols 3-4)
    # A[:, 0:3] * B[0:3, 3:5] -> C[:, 3:5]
    # U_sb_3x3x2 is (9, 18). Maps to A[i, j].
    # In full U, A[i, j] is at index i*4 + j.
    for i in range(3):
        for j in range(3):
            U_full[i*4 + j, r3+r15:r3+r15+r18] = U_sb_3x3x2[i*3 + j, :]
            
    # V_sb_3x3x2 is (9, 18). Maps to B[j, k] for j in 0..2, k in 3..4.
    # In full V, B[j, k] is at index j*5 + k.
    for j in range(3):
        for k in range(2):
            V_full[j*5 + (3+k), r3+r15:r3+r15+r18] = V_sb_3x3x2[j*3 + k, :]
            
    # W_sb_3x3x2 is (6, 18). Maps to C[k, i] for k in 3..4, i in 0..2.
    # In full W, C[k, i] is at index k*3 + i.
    for k in range(2):
        for i in range(3):
            W_full[(3+k)*3 + i, r3+r15:r3+r15+r18] = W_sb_3x3x2[k*3 + i, :]

    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Try to find 3x3x3 rank 23
    U3, V3, W3 = _find_3x3x3_decomposition(23, deadline - 10)
    
    if U3 is not None:
        # Get schoolbook for the other parts
        U_3x1x5, V_3x1x5, W_3x1x5 = _schoolbook(3, 1, 5)
        U_3x3x2, V_3x3x2, W_3x3x2 = _schoolbook(3, 3, 2)
        
        U_new, V_new, W_new = _build_3x4x5_from_3x3x3(
            np.array(U3), np.array(V3), np.array(W3),
            np.array(U_3x1x5), np.array(V_3x1x5), np.array(W_3x1x5),
            np.array(U_3x3x2), np.array(V_3x3x2), np.array(W_3x3x2)
        )
        
        if _verify(U_new, V_new, W_new, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = 23 + 15 + 18 # 56

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
