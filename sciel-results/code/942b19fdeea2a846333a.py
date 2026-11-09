import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Build target tensor T
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(dim_u, dim_v, dim_w) = 15
    # But likely higher. Start at 15, go up to 30.
    # We want the MINIMAL rank, so we check from low to high.
    
    # However, if minimal is 25, checking 15-24 might waste time.
    # But 15-24 are unlikely to have solutions.
    # Let's check 15 to 30.
    
    for R in range(15, 31):
        if time.time() > deadline - 10:
            break
            
        # Try to find decomposition of rank R
        found = _find_decomposition_als(T, dim_u, dim_v, dim_w, R, n, m, p, deadline)
        if found:
            return found
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_decomposition_als(T, dim_u, dim_v, dim_w, R, n, m, p, deadline):
    # ALS with rounding to half-integers
    # U: dim_u x R, V: dim_v x R, W: dim_w x R
    
    max_restarts = 10
    max_iters = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            return None
            
        # Initialize with small random half-integers
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_u, R))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_v, R))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_w, R))
        
        # Avoid zero matrices
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            # Check convergence
            T_decomp = _compute_tensor(U, V, W)
            if np.allclose(T_decomp, T, atol=1e-9):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
            # Update U
            U_new = np.zeros_like(U)
            for a in range(dim_u):
                # A_{(b,c), r} = V[b,r] * W[c,r]
                # We want to solve A @ U[a, :]^T = T[a, :, :]^T
                # Reshape T[a, :, :] to (dim_v * dim_w, 1)
                target = T[a, :, :].flatten()
                A = np.zeros((dim_v * dim_w, R))
                for b in range(dim_v):
                    for c in range(dim_w):
                        idx = b * dim_w + c
                        A[idx, :] = V[b, :] * W[c, :]
                
                # Solve least squares
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                    U_new[a, :] = np.round(sol * 2) / 2
                except:
                    U_new[a, :] = U[a, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(dim_v):
                target = T[:, b, :].flatten()
                A = np.zeros((dim_u * dim_w, R))
                for a in range(dim_u):
                    for c in range(dim_w):
                        idx = a * dim_w + c
                        A[idx, :] = U[a, :] * W[c, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                    V_new[b, :] = np.round(sol * 2) / 2
                except:
                    V_new[b, :] = V[b, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for c in range(dim_w):
                target = T[:, :, c].flatten()
                A = np.zeros((dim_u * dim_v, R))
                for a in range(dim_u):
                    for b in range(dim_v):
                        idx = a * dim_v + b
                        A[idx, :] = U[a, :] * V[b, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                    W_new[c, :] = np.round(sol * 2) / 2
                except:
                    W_new[c, :] = W[c, :]
            W = W_new
            
        # Final check after loop
        T_decomp = _compute_tensor(U, V, W)
        if np.allclose(T_decomp, T, atol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return None

def _compute_tensor(U, V, W):
    # U: dim_u x R, V: dim_v x R, W: dim_w x R
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    dim_u, R = U.shape
    dim_v, _ = V.shape
    dim_w, _ = W.shape
    
    T = np.zeros((dim_u, dim_v, dim_w))
    # Use einsum for speed
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return T

def _schoolbook(n, m, p):
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
