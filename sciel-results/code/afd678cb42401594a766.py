import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start = time.time()
    deadline = start + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # Build target tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Try ranks from lower bound up
    lower = max(n*m, m*p, n*p)  # 15
    upper = n*m*p  # 45
    
    best = None
    for R in range(lower, upper+1):
        if time.time() > deadline - 10:
            break
        # Try to find a rank-R decomposition
        found = try_rank(R, T, n, m, p)
        if found is not None:
            U, V, W = found
            best = (R, U, V, W)
            break  # found the smallest rank
        # if not found, continue to next R
    
    if best is None:
        # Fall back to naive
        R = upper
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best = (R, U, V, W)
    
    R, U, V, W = best
    # Convert to lists of integers/half-integers
    def to_list(M):
        return [[M[i,j] for j in range(M.shape[1])] for i in range(M.shape[0])]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

def try_rank(R, T, n, m, p):
    # ALS for the trilinear system
    best_residual = np.inf
    best_UVW = None
    nrestarts = 20
    for restart in range(nrestarts):
        # Random init
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        # ALS iterations
        for it in range(200):
            # Solve for U (row by row)
            for a in range(n*m):
                # For each (b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Linear in U[a,:]: sum_r (V[b,r]*W[c,r]) * U[a,r] = T[a,b,c]
                # Build matrix A[b,c,r] = V[b,r]*W[c,r], rhs[b,c] = T[a,b,c]
                A = (V[:, None, :] * W[None, :, :]).reshape(m*p*p*n, R)  # (b,c) -> r
                rhs = T[a, :, :].reshape(-1)
                # Solve least squares
                U[a, :], res, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            # Solve for V (row by row)
            for b in range(m*p):
                A = (U[:, None, :] * W[None, :, :]).reshape(n*m*p*n, R)  # (a,c) -> r
                rhs = T[:, b, :].reshape(-1)
                V[b, :], res, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            # Solve for W (row by row)
            for c in range(p*n):
                A = (U[:, None, :] * V[None, :, :]).reshape(n*m*m*p, R)  # (a,b) -> r
                rhs = T[:, :, c].reshape(-1)
                W[c, :], res, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            # Check residual
            residual = compute_residual(U, V, W, T)
            if residual < best_residual:
                best_residual = residual
                best_UVW = (U.copy(), V.copy(), W.copy())
            if residual < 1e-8:
                # Found an exact (or near-exact) solution
                # Try to snap to integers/half-integers
                U_s, V_s, W_s = snap_to_halfint(U, V, W)
                res_s = compute_residual(U_s, V_s, W_s, T)
                if res_s < 1e-6:
                    return (U_s, V_s, W_s)
                else:
                    # Snap didn't work; return the real solution? But it must be half-int.
                    # Let's try harder to snap
                    pass
    # If we found a near-exact real solution but not a half-integer one, 
    # we might still return the best real solution if it's close enough,
    # but the problem requires half-integers. So we need a half-integer solution.
    if best_residual < 1e-6 and best_UVW is not None:
        U, V, W = best_UVW
        U_s, V_s, W_s = snap_to_halfint(U, V, W)
        res_s = compute_residual(U_s, V_s, W_s, T)
        if res_s < 1e-6:
            return (U_s, V_s, W_s)
    return None

def compute_residual(U, V, W, T):
    # Compute the trilinear form and compare to T
    # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficiently: for each r, outer product
    result = np.zeros_like(T)
    for r in range(U.shape[1]):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.max(np.abs(result - T))

def snap_to_halfint(U, V, W):
    def snap(M):
        # Round to nearest half-integer
        return np.round(M * 2) / 2
    return snap(U), snap(V), snap(W)
