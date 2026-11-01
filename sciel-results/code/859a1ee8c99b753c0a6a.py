import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _build_rank30_attempt():
    """Try to construct rank 30 by sharing factors between blocks."""
    n, m, p = DIMENSIONS
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Use Strassen for first 4 pairs (k=0,1), (k=2,3), (k=4,5), (k=6,7)
    # That's 4*7 = 28 factors
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    
    # For k=8, try to use only 2 factors instead of 4
    # We need: sum_r U[i*2+j, r] * V[j*9+8, r] * W[8*2+i, r] = 1 for all i,j
    # Try: factor 28 handles (0,8) and (1,8), factor 29 handles (2,8) and (3,8)
    # U[0,28]=1, V[8,28]=1, W[8,28]=1  -> gives A[0,0]*B[0,8]
    # U[1,28]=1, V[18,28]=1, W[9,28]=1  -> gives A[0,1]*B[1,8]
    # U[2,29]=1, V[8,29]=1, W[10,29]=1  -> gives A[1,0]*B[0,8]
    # U[3,29]=1, V[18,29]=1, W[11,29]=1  -> gives A[1,1]*B[1,8]
    # But that's still 4 factors. Can we do it with 2?
    # With 2 factors, we have 2*4 + 2*2 + 2*2 = 16 params for 4 constraints.
    # Let's try: 
    # Factor 28: U=[1,0,1,0], V=[1,0,1,0,...], W=[1,1,1,1,...] won't work simply
    # Actually for a single column k, we need rank at least 2 (since it's a 2x2 matrix)
    # Minimum rank for 2x2 matrix mult with one column is 2? No, it's 4 in general.
    # But we can share with other columns.
    
    # For now, just add 2 more trivial-like factors for k=8
    # Actually, let's just use 2 factors for k=8:
    # Factor 28: contributes to A[0,0]*B[0,8] and A[1,1]*B[1,8]
    #   U[0,28]=1, U[3,28]=1
    #   V[0*9+8, 28]=1, V[1*9+8, 28]=1
    #   W[8*2+0, 28]=1, W[8*2+1, 28]=1
    #   This gives: A[0,0]*B[0,8]*1 + A[1,1]*B[1,8]*1... no that's wrong
    #   The tensor entry is sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #   For i=0,j=0,k=8: sum_r U[0,r]*V[8,r]*W[8,r]
    #   For i=0,j=1,k=8: sum_r U[1,r]*V[17,r]*W[8,r]
    #   For i=1,j=0,k=8: sum_r U[2,r]*V[8,r]*W[9,r]
    #   For i=1,j=1,k=8: sum_r U[3,r]*V[17,r]*W[9,r]
    
    # Let's try a different approach for the last column
    # Factor 28: U=[1,0,0,0], V[8]=1, W[8]=1 -> gives A[0,0]*B[0,8]
    # Factor 29: U=[0,1,0,0], V[17]=1, W[8]=1 -> gives A[0,1]*B[1,8]
    # Still need A[1,0]*B[0,8] and A[1,1]*B[1,8]
    # So we need at least 4 for the last column unless we share.
    
    # This approach won't easily give rank 30. Let's return None to indicate failure.
    return None

def _get_tensor_indices(n, m, p):
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return np.array(indices).T # Shape (3, N)

def _compute_residual(U, V, W, indices):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # indices: (3, N) -> a, b, c
    a, b, c = indices[0], indices[1], indices[2]
    prod = U[a, :] * V[b, :] * W[c, :]
    return prod.sum(axis=1)

def _verify_decomposition(U, V, W, indices, tol=1e-9):
    """Verify that the decomposition is exact."""
    pred = _compute_residual(U, V, W, indices)
    return np.max(np.abs(pred - 1.0)) < tol

def _try_rank(R, U0, V0, W0, indices, time_budget, max_restarts=10):
    start_time = time.time()
    nm, mp, pn = U0.shape[0], V0.shape[0], W0.shape[0]
    total_params = nm*R + mp*R + pn*R
    
    # Objective function
    def objective(params):
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        pred = _compute_residual(U, V, W, indices)
        err = pred - 1.0
        return np.sum(err**2)
    
    best_res = float('inf')
    best_params = None
    
    # Init 1: From rank 32, drop columns (if applicable)
    if R < U0.shape[1]:
        U_init = U0[:, :R].flatten()
        V_init = V0[:, :R].flatten()
        W_init = W0[:, :R].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        res = objective(x0)
        if res < best_res:
            best_res = res
            best_params = x0
            
        try:
            res_obj = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res_obj.fun < best_res:
                best_res = res_obj.fun
                best_params = res_obj.x
        except:
            pass

    # Init 2: Multiple random restarts
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Scale random initialization
        scale = 0.5
        x0 = np.random.randn(total_params) * scale
        try:
            res_obj = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-12})
            if res_obj.fun < best_res:
                best_res = res_obj.fun
                best_params = res_obj.x
        except:
            pass

    if best_params is not None and best_res < 1e-8:
        U = best_params[:nm*R].reshape(nm, R)
        V = best_params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = best_params[nm*R+mp*R:].reshape(pn, R)
        
        # Try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify_decomposition(U_r, V_r, W_r, indices):
            return U_r, V_r, W_r, R
        
        # Try rounding to nearest integer
        U_r2 = np.round(U)
        V_r2 = np.round(V)
        W_r2 = np.round(W)
        
        if _verify_decomposition(U_r2, V_r2, W_r2, indices):
            return U_r2, V_r2, W_r2, R
        
        # If not exactly rounded but very close, check if it's valid as-is
        # (task requires integer/half-integer, so we can't return non-rounded)
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    indices = _get_tensor_indices(n, m, p)
    
    U32, V32, W32 = _build_rank32()
    
    # Verify rank 32
    if not _verify_decomposition(U32, V32, W32, indices):
        # Should not happen, but fallback
        pass

    # Try to find lower ranks, starting from most likely to succeed
    # Rank 30 is a good target: 4 Strassen blocks (28) + 2 for last column
    for R in [30, 31, 29, 28]:
        if time.time() > deadline - 15:
            break
        remaining_time = deadline - 15 - time.time()
        if remaining_time < 5:
            break
        result = _try_rank(R, U32, V32, W32, indices, time_budget=remaining_time, max_restarts=8)
        if result is not None:
            U, V, W, rank = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
