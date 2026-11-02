import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Build the multiplication tensor T[a, b, c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for l in range(m):
            a = i * m + l
            for k in range(p):
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Known rank-21 solution
    U21, V21, W21 = _strassen_solution()
    
    best_U, best_V, best_W = U21, V21, W21
    best_rank = 21
    
    deadline = time.time() + 120
    
    for target_rank in range(20, 15, -1):
        if time.time() > deadline:
            break
        
        found = _find_decomposition(T, target_rank, deadline=deadline)
        if found is not None:
            U, V, W = found
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _strassen_solution():
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    
    return U, V, W

def _find_decomposition(T, target_rank, deadline=None):
    n_a, n_b, n_c = T.shape
    
    best_error = float('inf')
    best_result = None
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        U = np.random.randn(n_a, target_rank)
        V = np.random.randn(n_b, target_rank)
        W = np.random.randn(n_c, target_rank)
        
        for iteration in range(60):
            if deadline and time.time() > deadline:
                break
            
            VW = np.zeros((n_b * n_c, target_rank))
            for r in range(target_rank):
                VW[:, r] = (V[:, r] * W[:, r]).flatten()
            
            U_new = np.zeros((n_a, target_rank))
            for a in range(n_a):
                rhs = T[a, :, :].flatten()
                U_new[a, :], _, _, _ = np.linalg.lstsq(VW, rhs, rcond=None)
            U = U_new
            
            UW = np.zeros((n_a * n_c, target_rank))
            for r in range(target_rank):
                UW[:, r] = (U[:, r] * W[:, r]).flatten()
            
            V_new = np.zeros((n_b, target_rank))
            for b in range(n_b):
                rhs = T[:, b, :].flatten()
                V_new[b, :], _, _, _ = np.linalg.lstsq(UW, rhs, rcond=None)
            V = V_new
            
            UV = np.zeros((n_a * n_b, target_rank))
            for r in range(target_rank):
                UV[:, r] = (U[:, r] * V[:, r]).flatten()
            
            W_new = np.zeros((n_c, target_rank))
            for c in range(n_c):
                rhs = T[:, :, c].flatten()
                W_new[c, :], _, _, _ = np.linalg.lstsq(UV, rhs, rcond=None)
            W = W_new
        
        T_recon = np.zeros((n_a, n_b, n_c))
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        error = np.linalg.norm(T_recon - T)
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())
        
        if error < 1e-10:
            break
    
    if best_error < 1e-10:
        U, V, W = best_result
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_recon = np.zeros((n_a, n_b, n_c))
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
        
        if np.linalg.norm(T_recon - T) < 1e-10:
            return U_r, V_r, W_r
        
        # If rounding fails, the solution is not half-integer; don't use it
        return None
    return None
