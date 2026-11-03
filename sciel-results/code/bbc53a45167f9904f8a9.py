import time
import numpy as np
import scipy.linalg as la

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _als_optimize(U, V, W, n, m, p, max_iters=200, tol=1e-14):
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    r = U.shape[1]
    
    prev_err = float('inf')
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k] * W[k*n+i]
                b = np.ones(p)
                try:
                    sol, _, _, _ = la.lstsq(A.T, b, lapack_driver='gelsd')
                    U[i*m+j] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j] * W[k*n+i]
                b = np.ones(n)
                try:
                    sol, _, _, _ = la.lstsq(A.T, b, lapack_driver='gelsd')
                    V[j*p+k] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                b = np.ones(m)
                try:
                    sol, _, _, _ = la.lstsq(A.T, b, lapack_driver='gelsd')
                    W[k*n+i] = sol
                except:
                    pass
        
        new_err = _compute_residual(U, V, W, n, m, p)
        if abs(prev_err - new_err) < tol:
            break
        prev_err = new_err
    
    return U, V, W, _compute_residual(U, V, W, n, m, p)

def _snap_to_candidates(U, V, W, candidates):
    U_s = np.array(U)
    V_s = np.array(V)
    W_s = np.array(W)
    for mat in [U_s, V_s, W_s]:
        for i in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                mat[i, c] = candidates[np.argmin(np.abs(candidates - mat[i, c]))]
    return U_s, V_s, W_s

def _try_rank_r(n, m, p, r, max_restarts=20):
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    snap_candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Structured initialization: exploit cyclic structure
        U0 = np.zeros((n*m, r))
        V0 = np.zeros((m*p, r))
        W0 = np.zeros((p*n, r))
        
        for c in range(r):
            # Structured init exploiting 2-row symmetry
            # U: (6, r) -> rows 0-2 (i=0), rows 3-5 (i=1)
            base_u = np.random.choice([-1.0, 0.0, 1.0], size=3)
            U0[0:3, c] = base_u + np.random.uniform(-0.15, 0.15, 3)
            U0[3:6, c] = base_u * np.random.choice([-1.0, 1.0], size=3) + np.random.uniform(-0.15, 0.15, 3)
            
            # V: (18, r) -> for each k, 3 entries (j=0,1,2)
            for k in range(p):
                base_v = np.random.choice([-1.0, 0.0, 1.0], size=3)
                V0[k::p, c] = base_v + np.random.uniform(-0.15, 0.15, 3)
            
            # W: (12, r) -> for each k, 2 entries (i=0,1)
            for k in range(p):
                base_w = np.random.choice([-1.0, 0.0, 1.0], size=2)
                W0[k*2:k*2+2, c] = base_w + np.random.uniform(-0.15, 0.15, 2)
        
        # Run ALS
        U_als, V_als, W_als, err_als = _als_optimize(U0, V0, W0, n, m, p, max_iters=80)
        
        # Snap
        U_snap, V_snap, W_snap = _snap_to_candidates(U_als, V_als, W_als, snap_candidates)
        err_snap = _compute_residual(U_snap, V_snap, W_snap, n, m, p)
        
        if err_snap < 1e-8:
            # Polish
            U_pol, V_pol, W_pol, err_pol = _als_optimize(U_snap, V_snap, W_snap, n, m, p, max_iters=100, tol=1e-16)
            U_pol_s, V_pol_s, W_pol_s = _snap_to_candidates(U_pol, V_pol, W_pol, snap_candidates)
            err_pol_s = _compute_residual(U_pol_s, V_pol_s, W_pol_s, n, m, p)
            if err_pol_s < err_snap:
                U_snap, V_snap, W_snap = U_pol_s, V_pol_s, W_pol_s
                err_snap = err_pol_s
        
        if err_snap < best_err:
            best_err = err_snap
            best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
            if err_snap < 1e-12:
                return U_snap, V_snap, W_snap
    
    return (best_U, best_V, best_W) if best_U is not None else None

def construct():
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    for r in range(15, best_rank + 1):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        result = _try_rank_r(n, m, p, r, max_restarts=25)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_decomposition(U, V, W, n, m, p, tol=1e-9)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
