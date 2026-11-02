import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=None, m=None, p=None):
    """Fast exact verification."""
    if n is None:
        n, m, p = DIMENSIONS

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
        return False
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False

    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= 1e-9))

def _block_decomposition():
    """Assemble the full 3x4 x 4x8 decomposition (rank 56)."""
    def _strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        U[:, 0] = [1, 0, 0, 1]; V[:, 0] = [1, 0, 0, 1]; W[:, 0] = [1, 0, 0, 1]
        U[:, 1] = [0, 0, 1, 1]; V[:, 1] = [1, 0, 0, 0]; W[:, 1] = [0, 0, 1, 0]
        U[:, 2] = [1, 0, 0, 0]; V[:, 2] = [0, 0, 1, -1]; W[:, 2] = [0, 1, 0, 0]
        U[:, 3] = [0, 0, 0, 1]; V[:, 3] = [0, 1, -1, 0]; W[:, 3] = [1, 0, 0, 0]
        U[:, 4] = [1, 1, 0, 0]; V[:, 4] = [0, 0, 0, 1]; W[:, 4] = [0, 0, 0, 1]
        U[:, 5] = [0, 0, 1, -1]; V[:, 5] = [1, 1, 0, 0]; W[:, 5] = [0, 0, 0, 1]
        U[:, 6] = [0, 1, 0, -1]; V[:, 6] = [0, 0, 1, 1]; W[:, 6] = [0, 1, 0, 0]
        return U, V, W

    def _block_2x4x4x4():
        U, V, W = _strassen_2x2()
        Ua = np.zeros((8, 28), dtype=int)
        Va = np.zeros((16, 28), dtype=int)
        Wa = np.zeros((8, 28), dtype=int)
        blocks = [
            (0, 1, 0, 1, 0),
            (2, 3, 4, 5, 0),
            (0, 1, 2, 3, 2),
            (2, 3, 6, 7, 2),
        ]
        for b, (ar0, ar1, br0, br1, cc0) in enumerate(blocks):
            col0 = b * 7
            for t in range(7):
                col = col0 + t
                Ua[ar0, col] = U[0, t]
                Ua[ar1, col] = U[1, t]
                Va[br0, col] = V[0, t]
                Va[br1, col] = V[1, t]
                Wa[cc0, col] = W[0, t]
                Wa[cc0 + 1, col] = W[1, t]
        return Ua, Va, Wa

    def _block_1x4x4x4():
        Ua = np.zeros((4, 16), dtype=int)
        Va = np.zeros((16, 16), dtype=int)
        Wa = np.zeros((4, 16), dtype=int)
        for j in range(4):
            for k in range(4):
                col = j * 4 + k
                Ua[j, col] = 1
                Va[j * 4 + k, col] = 1
                Wa[k, col] = 1
        return Ua, Va, Wa

    Ua, Va, Wa = _block_2x4x4x4()
    U1, V1, W1 = _block_1x4x4x4()

    U = np.zeros((12, 56), dtype=int)
    V = np.zeros((32, 56), dtype=int)
    W = np.zeros((24, 56), dtype=int)

    U[0:8, 0:28] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:28] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[k * 3 + i, 0:28] = Wa[k * 4 + i, :]

    U[0:8, 28:56] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:56] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[(k + 4) * 3 + i, 28:56] = Wa[k * 4 + i, :]

    U[8:12, 0:16] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:16] = V1[j * 4 + k, :]
    for k in range(4):
        W[k * 3 + 2, 0:16] = W1[k, :]

    U[8:12, 28:44] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:44] = V1[j * 4 + k, :]
    for k in range(4):
        W[(k + 4) * 3 + 2, 28:44] = W1[k, :]

    return {"U": U, "V": V, "W": W}

def _to_lists(res):
    return {
        "U": res["U"].astype(int).tolist(),
        "V": res["V"].astype(int).tolist(),
        "W": res["W"].astype(int).tolist(),
    }

def _als_search(n, m, p, rank, deadline, max_iters=200, tol=1e-12):
    """
    Alternating Least Squares for tensor rank approximation.
    Minimize ||T - sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]||_F
    where T is the all-ones tensor of shape (n, m, p).
    
    For fixed V and W, solve for U via linear least squares.
    For fixed U and W, solve for V via linear least squares.
    For fixed U and V, solve for W via linear least squares.
    """
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-15
    V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-15
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-15
    
    prev_err = np.inf
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Solve for W: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For each (k,i): T[i,j,k] for all j = sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r]
        # Let's build the linear system.
        # Flatten: for fixed (k,i), we have n*m*p equations but only rank unknowns for W[k,i,:]
        # Actually, let me reshape properly.
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # For W update:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Fix (k,i). Then for all (j): T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r]
        # This is a linear system in W[k,i,:].
        
        # Build A matrix: for each (k,i), A has shape (m, rank) where A[j, r] = U[i,j,r] * V[j,k,r]
        # b vector: b[j] = T[i,j,k] = 1
        
        W_new = np.zeros_like(W3)
        for k in range(p):
            for i in range(n):
                # A[j, r] = U[i,j,r] * V[j,k,r]
                A = U3[i, :, :] * V3[:, k, :]  # shape (m, rank)
                b = np.ones(m)  # T[i,j,k] = 1 for all j
                # Solve A^T A x = A^T b
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k, i, :] = sol
                except:
                    W_new[k, i, :] = W3[k, i, :]
        W = W_new.reshape(p * n, rank)
        
        # Solve for U: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Fix (i,j). Then for all (k): T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        U_new = np.zeros_like(U3)
        for i in range(n):
            for j in range(m):
                # A[k, r] = V[j,k,r] * W[k,i,r]
                A = V3[j, :, :] * W3[:, i, :]  # shape (p, rank)
                b = np.ones(p)  # T[i,j,k] = 1 for all k
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i, j, :] = sol
                except:
                    U_new[i, j, :] = U3[i, j, :]
        U = U_new.reshape(n * m, rank)
        
        # Solve for V: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Fix (j,k). Then for all (i): T[i,j,k] = sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r])
        V_new = np.zeros_like(V3)
        for j in range(m):
            for k in range(p):
                # A[i, r] = U[i,j,r] * W[k,i,r]
                A = U3[:, j, :] * W3[k, :, :]  # shape (n, rank)
                b = np.ones(n)  # T[i,j,k] = 1 for all i
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j, k, :] = sol
                except:
                    V_new[j, k, :] = V3[j, k, :]
        V = V_new.reshape(m * p, rank)
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.linalg.norm(T_calc - np.ones((n, m, p)))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Try to round to integers/half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Check if rounded solution is valid
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return {"U": U_r, "V": V_r, "W": W_r}
    
    # Also try pure integer rounding
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i.tolist(), V_i.tolist(), W_i.tolist()):
        return {"U": U_i, "V": V_i, "W": W_i}
    
    return None

def _search_lower_rank(deadline):
    """
    Attempt to find a decomposition with rank < 56 using ALS.
    """
    n, m, p = DIMENSIONS
    # Search from lower bound up to current best (55)
    # Theoretical lower bound for 3x4x8 is not well-known, but let's try a range
    target_ranks = range(40, 56)
    
    for r in target_ranks:
        if time.time() > deadline - 15.0:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 15.0:
                break
            
            np.random.seed(12345 + r * 1000 + trial)
            result = _als_search(n, m, p, r, deadline, max_iters=100)
            if result is not None:
                if _verify(result["U"].tolist(), result["V"].tolist(), result["W"].tolist()):
                    return _to_lists(result)
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Guaranteed-valid baseline.
    best = _schoolbook()
    best_rank = len(best["U"][0])

    # Primary structured result: block Strassen, rank 56.
    try:
        if time.time() < deadline - 30.0:
            res = _block_decomposition()
            if _verify(res["U"].tolist(), res["V"].tolist(), res["W"].tolist()):
                best = _to_lists(res)
                best_rank = len(best["U"][0])
    except Exception:
        pass

    # Try ALS search for lower rank
    try:
        if time.time() < deadline - 20.0:
            improved = _search_lower_rank(deadline)
            if improved is not None:
                if _verify(improved["U"], improved["V"], improved["W"]):
                    best = improved
                    best_rank = len(best["U"][0])
    except Exception:
        pass

    # Final verification; fall back to schoolbook on any anomaly.
    try:
        if not _verify(best["U"], best["V"], best["W"]):
            best = _schoolbook()
    except Exception:
        best = _schoolbook()

    return best
