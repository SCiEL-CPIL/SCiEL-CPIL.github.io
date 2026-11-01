import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Fallback solution: Rank 21 (3 independent Strassen blocks)
    rank_fb = 21
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]

    U_fb = [[0] * rank_fb for _ in range(4)]
    V_fb = [[0] * rank_fb for _ in range(12)]
    W_fb = [[0] * rank_fb for _ in range(12)]

    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U_fb[idx][col] = sc['U'][idx]
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V_fb[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W_fb[w_idx[idx]][col] = sc['W'][idx]

    # Build target tensor T for M_{2,2,6}
    # A index a = i*2 + j (0..3)
    # B index b = j*6 + k (0..11)
    # C index c = k*2 + i (0..11)  [W[k*n+i, r] convention]
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T[a, b, c] = 1.0

    target_rank = 20
    dim_U = 4
    dim_V = 12
    dim_W = 12
    total_params = target_rank * (dim_U + dim_V + dim_W)  # 20 * 28 = 560

    def residuals(params):
        U_s = params[0:target_rank * dim_U].reshape(target_rank, dim_U)
        V_s = params[target_rank * dim_U:target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V)
        W_s = params[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W)
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        return (S - T).flatten()

    def try_solve(init, max_nfev=3000, tol=1e-10):
        """Try to solve from given initialization. Returns (success, params, cost)."""
        if time.time() > deadline - 10:
            return False, None, np.inf
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=max_nfev, ftol=tol, xtol=tol, gtol=tol)
            cost = result.cost
            if cost < 1e-8:
                return True, result.x, cost
            return False, result.x, cost
        except Exception:
            return False, None, np.inf

    def params_to_matrices(params):
        U_s = params[0:target_rank * dim_U].reshape(target_rank, dim_U)
        V_s = params[target_rank * dim_U:target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V)
        W_s = params[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W)
        U = [[0] * target_rank for _ in range(dim_U)]
        V = [[0] * target_rank for _ in range(dim_V)]
        W = [[0] * target_rank for _ in range(dim_W)]
        for s in range(target_rank):
            for idx in range(dim_U):
                U[idx][s] = U_s[s, idx]
            for idx in range(dim_V):
                V[idx][s] = V_s[s, idx]
            for idx in range(dim_W):
                W[idx][s] = W_s[s, idx]
        return U, V, W

    # Strategy: Use the rank-21 Strassen solution as a warm start.
    # Project it to rank 20 by finding the best rank-20 approximation,
    # then refine.
    
    np.random.seed(42)
    
    # Build the rank-21 factor matrices as numpy arrays
    U21 = np.array(U_fb, dtype=float)  # (4, 21)
    V21 = np.array(V_fb, dtype=float)  # (12, 21)
    W21 = np.array(W_fb, dtype=float)  # (12, 21)
    
    # The rank-21 solution exactly represents T. We want to find a rank-20
    # decomposition. One approach: use the structure that A is shared.
    # 
    # Alternative: Try to find a rank-20 decomposition directly with 
    # structured initializations.
    
    # Initialization strategy 1: Random
    found = False
    best_params = None
    best_cost = np.inf
    
    # Try several random initializations
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        init = np.random.randn(total_params) * 0.1
        success, params, cost = try_solve(init, max_nfev=2000)
        if cost < best_cost:
            best_cost = cost
            best_params = params
        if success:
            found = True
            best_params = params
            break
    
    # Initialization strategy 2: Based on rank-21 solution
    # Take the rank-21 solution and try to merge two columns
    # The idea: if two Strassen steps across different blocks can be combined
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        
        # Create a rank-20 initialization by perturbing the rank-21 structure
        # Take 20 out of 21 columns with small perturbation
        cols_to_keep = list(range(21))
        # Remove one column (randomly choose which)
        remove_col = np.random.randint(21)
        cols_to_keep.remove(remove_col)
        
        # Build initial params from kept columns
        init = np.zeros(total_params)
        for new_col, old_col in enumerate(cols_to_keep):
            init[new_col * dim_U:(new_col + 1) * dim_U] = U21[:, old_col]
            init[(new_col + 1) * dim_U:(new_col + 1) * dim_U + dim_V] = V21[:, old_col]
            init[(new_col + 1) * dim_U + dim_V:(new_col + 2) * dim_U + dim_V] = W21[:, old_col]
        
        # Add small random perturbation
        init += np.random.randn(total_params) * 0.05
        
        success, params, cost = try_solve(init, max_nfev=3000)
        if cost < best_cost:
            best_cost = cost
            best_params = params
        if success:
            found = True
            best_params = params
            break
    
    # Initialization strategy 3: Use SVD-like compression of the rank-21 solution
    # The tensor T has a rank-21 decomposition. We can try to find its
    # best rank-20 approximation using iterative methods.
    for trial in range(10):
        if time.time() > deadline - 15:
            break
        
        # Start from a random rank-20 and do a few steps of ALS
        U_s = np.random.randn(target_rank, dim_U) * 0.1
        V_s = np.random.randn(target_rank, dim_V) * 0.1
        W_s = np.random.randn(target_rank, dim_W) * 0.1
        
        # Do a few ALS iterations to get closer to a solution
        for it in range(50):
            if time.time() > deadline - 15:
                break
            # Update U: for each a, U[:, a] = argmin ||T[a] - V_s @ W_s^T * U[:, a]||
            # Actually, let's just use least_squares from this point
            break
        
        init = np.concatenate([U_s.flatten(), V_s.flatten(), W_s.flatten()])
        success, params, cost = try_solve(init, max_nfev=2000)
        if cost < best_cost:
            best_cost = cost
            best_params = params
        if success:
            found = True
            best_params = params
            break

    # If we found a solution, verify and return
    if found and best_params is not None:
        # Do a final polish with tighter tolerance
        if time.time() < deadline - 5:
            try:
                result = least_squares(residuals, best_params, method='lm', max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if result.cost < 1e-8:
                    best_params = result.x
            except Exception:
                pass
        
        U, V, W = params_to_matrices(best_params)
        
        # Verify
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        S = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if np.max(np.abs(S - T)) > 1e-6:
            # Verification failed, return fallback
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
