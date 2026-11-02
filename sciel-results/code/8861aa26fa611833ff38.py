import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find lower rank
    for rank in range(27, rank_school):
        if time.time() > deadline - 20:
            break
            
        result = _find_decomposition_als(n, m, p, rank, deadline)
        if result is not None:
            return result
            
    return _schoolbook(n, m, p)

def _find_decomposition_als(n, m, p, rank, deadline):
    # Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i'
    
    # Precompute indices
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # We need to solve linear systems.
    # For U: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[i*m+j, j'*p+k, k'*n+i']
    # This is linear in U.
    # Let's flatten V and W to form the design matrix.
    
    # Actually, it's easier to just use the structure.
    # For a fixed i, j, the equation for U[i*m+j, :] is:
    # sum_r U[i*m+j, r] * (V[j'*p+k, r] * W[k'*n+i', r]) = T[i*m+j, j'*p+k, k'*n+i']
    # Note that T is non-zero only when j'=j, k'=k, i'=i.
    # So for a specific (i,j), the non-zero entries in T are for (j,k,i).
    # Wait, T is sparse.
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i).
    # So for a fixed U index u=(i,j), the non-zero T entries are v=(j,k) for all k, and w=(k,i) for those k.
    # There are p such entries.
    
    # So for each u, we have p equations.
    # sum_r U[u, r] * V[(j,k)*p + k, r] * W[(k,i)*n + i, r] = 1 for k=0..p-1.
    
    # This is a system of p equations with rank unknowns.
    # If rank > p, it's underdetermined.
    
    # We can solve for U given V, W.
    # Let's build the matrix A_u of shape (p, rank).
    # A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
    # Then A_u @ U[u, :] = ones(p)
    
    # Similarly for V and W.
    
    # Initialize V, W randomly
    np.random.seed(42)
    
    best_residual = float('inf')
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
            
        V = np.random.randn(V_size, rank)
        W = np.random.randn(W_size, rank)
        
        for it in range(20):
            if time.time() > deadline - 5:
                break
                
            # Solve for U
            U = np.zeros((U_size, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_u = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        A_u[k, :] = V[v_idx, :] * W[w_idx, :]
                    # Solve A_u @ U[u, :] = ones(p)
                    try:
                        U[u, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
                    except:
                        continue
                        
            # Solve for V
            V_new = np.zeros((V_size, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_v = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        A_v[i, :] = U[u_idx, :] * W[w_idx, :]
                    # Solve A_v @ V[v, :] = ones(n)
                    try:
                        V_new[v, :], residuals, rank_v, sv = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
                    except:
                        continue
            V = V_new
            
            # Solve for W
            W_new = np.zeros((W_size, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_w = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A_w[j, :] = U[u_idx, :] * V[v_idx, :]
                    # Solve A_w @ W[w, :] = ones(m)
                    try:
                        W_new[w, :], residuals, rank_w, sv = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                    except:
                        continue
            W = W_new
            
            # Check residual
            res = _compute_residual(U, V, W, n, m, p)
            residual_norm = np.max(np.abs(res))
            
            if residual_norm < 1e-8:
                # Try to snap
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                res_snap = _compute_residual(U_snap, V_snap, W_snap, n, m, p)
                if np.max(np.abs(res_snap)) < 1e-6:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                else:
                    # If snap fails, maybe return float? 
                    # Prompt says "integer/half-integer entries". 
                    # If we can't snap, we can't return this.
                    pass
                    
    return None

def _compute_residual(U, V, W, n, m, p):
    res = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]) - 1.0
    return res

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
