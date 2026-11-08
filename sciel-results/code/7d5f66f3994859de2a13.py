import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: Find U (n*m, R), V (m*p, R), W (p*n, R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Known upper bound: 28 (4 blocks of Strassen 2x2x2)
    # Lower bound: 16 (trivial)
    # We try to find a solution with R < 28.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all valid (u_idx, v_idx, w_idx) triples
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    # Precompute index arrays for vectorized evaluation
    u_idxs = np.array([t[0] for t in indices])
    v_idxs = np.array([t[1] for t in indices])
    w_idxs = np.array([t[2] for t in indices])
    
    def evaluate_solution(U, V, W):
        """
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Returns max absolute error.
        """
        R = U.shape[1]
        errors = np.zeros(len(indices))
        for r in range(R):
            # Vectorized dot product for all indices at rank r
            # U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            vals = U[u_idxs, r] * V[v_idxs, r] * W[w_idxs, r]
            errors += vals
        return np.max(np.abs(errors - 1.0))
    
    def round_to_half(U, V, W):
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        return U_r, V_r, W_r
    
    # Strategy:
    # 1. Start with the known rank 28 solution (Strassen blocks).
    # 2. Try to optimize a rank 27 solution initialized from rank 28 by zeroing a column and re-optimizing?
    #    Or better: Random restarts for R = 27, 26, ... using BFGS or L-BFGS-B.
    #    Since the problem is non-convex, random restarts are key.
    
    # First, construct the rank 28 solution to have a fallback.
    def construct_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff
                
        def strassen_terms():
            terms = []
            # M1
            u = {0: 1, 3: 1}
            v = {0: 1, 3: 1}
            w = {0: 1, 3: 1}
            terms.append((u, v, w))
            # M2
            u = {2: 1, 3: 1}
            v = {0: 1}
            w = {2: 1, 3: -1}
            terms.append((u, v, w))
            # M3
            u = {0: 1}
            v = {1: 1, 3: -1}
            w = {1: 1, 3: 1}
            terms.append((u, v, w))
            # M4
            u = {3: 1}
            v = {2: 1, 0: -1}
            w = {0: 1, 2: 1}
            terms.append((u, v, w))
            # M5
            u = {0: 1, 1: 1}
            v = {3: 1}
            w = {0: -1, 1: 1}
            terms.append((u, v, w))
            # M6
            u = {2: 1, 0: -1}
            v = {0: 1, 1: 1}
            w = {3: 1}
            terms.append((u, v, w))
            # M7
            u = {1: 1, 3: -1}
            v = {2: 1, 3: 1}
            w = {0: 1}
            terms.append((u, v, w))
            return terms

        base_terms = strassen_terms()
        
        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx
                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff
                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff
                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff
                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)
        
        return U, V, W

    U_28, V_28, W_28 = construct_rank_28()
    # Verify rank 28 is valid (should be)
    err_28 = evaluate_solution(U_28, V_28, W_28)
    if err_28 > 1e-9:
        # Fallback to identity if construction failed (unlikely)
        pass

    best_rank = 28
    best_U, best_V, best_W = U_28.copy(), V_28.copy(), W_28.copy()
    
    # Search for lower ranks
    # Use scipy.optimize if available, otherwise simple gradient descent or random search.
    # Since we have time budget, let's try BFGS on a subset of ranks.
    
    try:
        import scipy.optimize as opt
        has_scipy = True
    except ImportError:
        has_scipy = False

    # Objective function for optimization
    # Variables: flattened U, V, W
    def objective(params, R):
        U_flat = params[:n*m*R]
        V_flat = params[n*m*R : n*m*R + m*p*R]
        W_flat = params[n*m*R + m*p*R :]
        
        U_f = U_flat.reshape(n*m, R)
        V_f = V_flat.reshape(m*p, R)
        W_f = W_flat.reshape(p*n, R)
        
        # Compute errors
        errors = np.zeros(len(indices))
        for r in range(R):
            vals = U_f[u_idxs, r] * V_f[v_idxs, r] * W_f[w_idxs, r]
            errors += vals
        
        # Sum of squares of errors
        return np.sum((errors - 1.0) ** 2)

    # Try ranks from 27 down to 20
    for R in range(27, 19, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on time remaining
        max_restarts = 10 if R > 24 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Random sparse initialization might help
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            # Add a small component from the rank 28 solution to guide?
            # Or just random.
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            if has_scipy:
                try:
                    # Use L-BFGS-B for better handling of large variable counts
                    res = opt.minimize(objective, params_init, args=(R,), method='L-BFGS-B',
                                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                    if res.fun < 1e-6:
                        # Recover
                        U_f = res.x[:n*m*R].reshape(n*m, R)
                        V_f = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                        W_f = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                        
                        # Round to half-integers
                        U_r, V_r, W_r = round_to_half(U_f, V_f, W_f)
                        
                        # Verify rounded
                        err = evaluate_solution(U_r, V_r, W_r)
                        if err < 1e-9:
                            best_rank = R
                            best_U, best_V, best_W = U_r, V_r, W_r
                            found = True
                            break
                except Exception:
                    pass
            else:
                # Fallback: simple random search with local refinement?
                # Too slow. Just skip if no scipy.
                pass
        
        if found:
            # Found a lower rank, continue searching for even lower?
            # The loop continues to lower R.
            pass
        else:
            # If we fail to find R, it's unlikely to find R-1 easily?
            # But maybe a different random seed works.
            # Let's keep searching lower ranks as well, just with fewer restarts if time is tight.
            pass

    # Final check on the best result
    if evaluate_solution(best_U, best_V, best_W) < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return the known good rank 28
        return {"U": U_28.tolist(), "V": V_28.tolist(), "W": W_28.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
