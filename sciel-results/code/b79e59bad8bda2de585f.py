"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def tensor_error(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T))
    
    def polish(U, V, W, max_iters=5000, tol=1e-15):
        """High-precision polish pass."""
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            TW = np.einsum('abc,cr->abr', T, W)
            inner_U = np.einsum('abr,br->ar', TW, V)
            denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
            U_new = inner_U / (denom_U + 1e-30)
            
            TU = np.einsum('abc,ar->bcr', T, U_new)
            inner_V = np.einsum('bcr,cr->br', TU, W)
            denom_V = np.sum(U_new**2, axis=0) * np.sum(W**2, axis=0)
            V_new = inner_V / (denom_V + 1e-30)
            
            inner_W = np.einsum('bcr,br->cr', TU, V_new)
            denom_W = np.sum(U_new**2, axis=0) * np.sum(V_new**2, axis=0)
            W_new = inner_W / (denom_W + 1e-30)
            
            U, V, W = U_new, V_new, W_new
            
            if it % 50 == 0 or it == max_iters - 1:
                err = tensor_error(U, V, W)
                if err < tol:
                    break
        
        return U, V, W, tensor_error(U, V, W)
    
    def build_rank21():
        rank = 21
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        
        W_coeffs = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        for b in range(3):
            for t in range(7):
                r = b * 7 + t
                for idx, val in U_coeffs[t].items():
                    U[idx, r] = val
                for (j, k_local, val) in V_coeffs[t]:
                    k_global = 2 * b + k_local
                    v_idx = j * p + k_global
                    V[v_idx, r] = val
                for (i, k_local, val) in W_coeffs[t]:
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx, r] = val
        
        return U, V, W
    
    def try_rank20():
        """Try to find rank-20 by perturbing rank-21 construction."""
        U21, V21, W21 = build_rank21()
        rank = 20
        best_err = float('inf')
        best_result = None
        
        for seed in range(40):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.RandomState(seed)
            
            # Strategy: start from rank-21, drop one term, perturb rest
            drop_idx = seed % 21
            U = U21.copy()
            V = V21.copy()
            W = W21.copy()
            
            U[:, drop_idx] = 0
            V[:, drop_idx] = 0
            W[:, drop_idx] = 0
            
            for r in range(21):
                if r == drop_idx:
                    continue
                U[:, r] += rng.randn(n*m) * 0.05
                V[:, r] += rng.randn(m*p) * 0.05
                W[:, r] += rng.randn(p*n) * 0.05
            
            # Keep first 20 columns
            U = U[:, :20].copy()
            V = V[:, :20].copy()
            W = W[:, :20].copy()
            
            # ALS
            for it in range(150):
                if time.time() > deadline - 30:
                    break
                
                TW = np.einsum('abc,cr->abr', T, W)
                inner_U = np.einsum('abr,br->ar', TW, V)
                denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
                U = inner_U / (denom_U + 1e-15)
                
                TU = np.einsum('abc,ar->bcr', T, U)
                inner_V = np.einsum('bcr,cr->br', TU, W)
                denom_V = np.sum(U**2, axis=0) * np.sum(W**2, axis=0)
                V = inner_V / (denom_V + 1e-15)
                
                inner_W = np.einsum('bcr,br->cr', TU, V)
                denom_W = np.sum(U**2, axis=0) * np.sum(V**2, axis=0)
                W = inner_W / (denom_W + 1e-15)
                
                if it % 15 == 0 or it == 149:
                    err = tensor_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    if err < 1e-12:
                        break
            
            if best_err < 1e-10:
                break
        
        if best_result is not None and best_err < 1e-10:
            U, V, W = best_result
            U_p, V_p, W_p, err_p = polish(U, V, W)
            if err_p < 1e-12:
                return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
        
        return None
    
    # Try rank-20
    result20 = try_rank20()
    if result20 is not None:
        return result20
    
    # Fall back to rank-21
    U, V, W = build_rank21()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
