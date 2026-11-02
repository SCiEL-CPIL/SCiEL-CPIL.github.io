import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p
    
    # Precompute indices
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(res, np.ones(total_entries), atol=tol, rtol=0)

    def get_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W

    def try_find_rank(R, time_limit=10.0, seed=0):
        """
        Attempt to find a rank R decomposition using least squares and rounding.
        """
        if R >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        
        # We use multiple restarts
        num_restarts = 5
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small random values
            # Scale matters for convergence
            U = rng.uniform(-1, 1, (nm, R))
            V = rng.uniform(-1, 1, (mp, R))
            W = rng.uniform(-1, 1, (pn, R))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R+mp*R]
                W_flat = x[nm*R+mp*R:]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                U_sel = U_mat[idx_u, :]
                V_sel = V_mat[idx_v, :]
                W_sel = W_mat[idx_w, :]
                
                res = np.sum(U_sel * V_sel * W_sel, axis=1)
                return res - 1.0
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-12, ftol=1e-12)
                
                if result.success or result.cost < 1e-10:
                    x_sol = result.x
                    U_sol = x_sol[:nm*R].reshape(nm, R)
                    V_sol = x_sol[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_sol = x_sol[nm*R+mp*R:].reshape(pn, R)
                    
                    # Check if it's already close to half-integers
                    # Round to nearest half-integer
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, try local search
                    # We check if the error is small enough to warrant a local search
                    if result.cost < 1e-6:
                        # Local search: perturb entries to nearest half-integers
                        # This is a simplified local search
                        current_U, current_V, current_W = U_r, V_r, W_r
                        current_err = np.sum((np.sum(current_U[idx_u, :] * current_V[idx_v, :] * current_W[idx_w, :], axis=1) - 1)**2)
                        
                        improved = True
                        while improved and (time.time() - start_time < time_limit):
                            improved = False
                            # Try flipping signs or adding 0.5 to random entries?
                            # Simpler: check all entries if changing to nearest half-integer (other than current) helps
                            # But that's expensive. 
                            # Given the structure, if LS found a solution, it's likely very close to a grid point.
                            # If rounding failed, maybe the solution is not on the grid?
                            # But the problem states integer/half-integer entries EXIST for the optimal rank.
                            # So if LS converges, the grid point should be nearby.
                            
                            # Let's try a few random perturbations
                            for _ in range(10):
                                U_p = current_U.copy()
                                V_p = current_V.copy()
                                W_p = current_W.copy()
                                
                                # Perturb a few entries
                                for _ in range(3):
                                    idx = rng.randint(nm*R)
                                    if idx < nm*R:
                                        r_idx, c_idx = np.unravel_index(idx, (nm, R))
                                        U_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                    elif idx < nm*R + mp*R:
                                        idx_v = idx - nm*R
                                        r_idx, c_idx = np.unravel_index(idx_v, (mp, R))
                                        V_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                    else:
                                        idx_w = idx - nm*R - mp*R
                                        r_idx, c_idx = np.unravel_index(idx_w, (pn, R))
                                        W_p[r_idx, c_idx] += rng.choice([-0.5, 0.5])
                                        
                                if verify_solution(U_p, V_p, W_p):
                                    return U_p, V_p, W_p
                                    
            except Exception as e:
                pass
                
        return None

    # Main Logic
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    timeout = 110.0
    
    # Try ranks from 29 down to 25
    for R in range(29, 24, -1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        for seed in range(10):
            if time.time() - start_time > timeout:
                break
            res = try_find_rank(R, time_limit=2.0, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            # If we found a solution for R, we can try to go lower
            # But if we find 29, we should try 28, etc.
            continue
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
