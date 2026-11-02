import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 20
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                l = k*n + i
                T[u, v, l] = 1.0
    
    T_flat = T.flatten()
    
    def get_rank_21():
        # ... same as before ...
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
                    U[idx][r] = val
                for l_idx, val in v_coeffs.items():
                    if l_idx == 0: v_idx = 0 * p + 2 * b
                    elif l_idx == 1: v_idx = 0 * p + 2 * b + 1
                    elif l_idx == 2: v_idx = 1 * p + 2 * b
                    elif l_idx == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i_idx, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i_idx
                    W[w_idx][r] = val
                r += 1
        return U, V, W

    U21, V21, W21 = get_rank_21()
    
    def try_rank_20():
        R = target_rank
        size_u = n*m*R
        size_v = m*p*R
        
        def residual(x):
            U = x[:size_u].reshape(n*m, R)
            V = x[size_u:size_u+size_v].reshape(m*p, R)
            W = x[size_u+size_v:].reshape(p*n, R)
            
            # Compute tensor D
            # D[u, v, l] = sum_r U[u, r] V[v, r] W[l, r]
            D = np.einsum('ur,vr,lr->uvl', U, V, W)
            return D.flatten() - T_flat
        
        # SVD-based initial guess: For each basis element of A (index u), 
        # the slice T[u, :, :] is a (m*p) x (p*n) matrix.
        # The rank of T is at most the number of non-zero singular values.
        # We use this to construct a better initial guess.
        
        # First, try the simple truncation approach
        U_init = U21[:, :R].flatten()
        V_init = V21[:, :R].flatten()
        W_init = W21[:, :R].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        best_result = None
        best_cost = np.inf
        
        # Try multiple initializations
        initializations = []
        
        # Initialization 1: Truncation of rank-21
        initializations.append(x0)
        
        # Initialization 2: SVD-based
        # For the multiplication tensor, we can think of it as a set of linear maps.
        # Reshape T to a matrix and use SVD.
        # T has shape (n*m, m*p, p*n) = (4, 12, 12)
        # We can flatten the last two dimensions: T_mat has shape (4, 144)
        T_mat = T.reshape(n*m, -1)
        U_svd, S_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        
        # Use top R singular vectors
        if S_vals.size >= R:
            U_svd_top = U_svd[:, :R]
            S_top = S_vals[:R]
            Vt_top = Vt_svd[:R, :]
            
            # Construct initial guess from SVD
            # U[u, r] = U_svd_top[u, r] / sqrt(S_top[r])
            # V and W need to be extracted from Vt_top
            # Vt_top has shape (R, m*p * p*n) = (R, 144)
            # We need to split this into V and W parts
            
            # Actually, for matrix multiplication tensor, the structure is more specific.
            # Let's use a different approach: use the rank-21 solution but perturb it.
            pass
        
        # Initialization 3: Random perturbation of rank-21 truncation
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            U_pert = U21[:, :R] + 0.1 * np.random.randn(n*m, R)
            V_pert = V21[:, :R] + 0.1 * np.random.randn(m*p, R)
            W_pert = W21[:, :R] + 0.1 * np.random.randn(p*n, R)
            x_pert = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
            initializations.append(x_pert)
        
        # Try each initialization
        for x0_trial in initializations:
            if time.time() > deadline - 10:
                break
            try:
                res = least_squares(residual, x0_trial, method='lm', max_nfev=15000)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res.x
                    if res.cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_result is not None and best_cost < 1e-8:
            U_opt = best_result[:size_u].reshape(n*m, R)
            V_opt = best_result[size_u:size_u+size_v].reshape(m*p, R)
            W_opt = best_result[size_u+size_v:].reshape(p*n, R)
            
            # Round to nearest 0.5
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            # Verify
            D_round = np.einsum('ur,vr,lr->uvl', U_round, V_round, W_round)
            if np.allclose(D_round.flatten(), T_flat, atol=1e-9):
                return U_round, V_round, W_round
            
            # Try rounding with different strategies
            for scale in [1, 2, 4]:
                U_round2 = np.round(U_opt * scale) / scale
                V_round2 = np.round(V_opt * scale) / scale
                W_round2 = np.round(W_opt * scale) / scale
                
                D_round2 = np.einsum('ur,vr,lr->uvl', U_round2, V_round2, W_round2)
                if np.allclose(D_round2.flatten(), T_flat, atol=1e-9):
                    return U_round2, V_round2, W_round2
        
        return None

    # Try rank 20
    if time.time() < deadline:
        result = try_rank_20()
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 21
    U, V, W = get_rank_21()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
