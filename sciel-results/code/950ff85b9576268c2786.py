import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s margin before 150s kill

    def _verify(U, V, W):
        """Verify exact decomposition. U: (nm, R), V: (mp, R), W: (pn, R)
        Convention: W[k*n + i, r] where i is row of A, j is col of A/row of B, k is col of B.
        S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
        """
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return (U, V, W)

    def _snap_to_half(params, shape_list):
        """Snap parameters to nearest 0.5, verify, return (U,V,W) or None."""
        pos = 0
        arrays = []
        for shape in shape_list:
            sz = 1
            for d in shape:
                sz *= d
            arr = params[pos:pos + sz].reshape(shape)
            arr_snap = np.round(arr * 2) / 2
            arrays.append(arr_snap)
            pos += sz
        U, V, W = arrays
        if _verify(U, V, W):
            return (U, V, W)
        return None

    def _residual(params):
        R = len(params) // (n * m + m * p + p * n)
        if R == 0:
            return np.zeros(1)
        U = params[0:n * m * R].reshape(n * m, R)
        V = params[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
        W = params[n * m * R + m * p * R:].reshape(p * n, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return S.flatten() - 1.0

    def _try_rank(R, n_restarts=8, method='lm'):
        num_vars = R * (n * m + m * p + p * n)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(restart * 7919 + R * 13 + 42)
            
            # Initialize with random values scaled to help convergence
            scale = 0.7
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Gauge fixing: set first row of U to 1 to fix scale ambiguity
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import least_squares
                if method == 'lm':
                    sol = least_squares(_residual, params, method='lm', max_nfev=2000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
                else:
                    sol = least_squares(_residual, params, method='trf', max_nfev=5000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            except Exception:
                continue
            
            if np.linalg.norm(sol.fun) < 1e-7:
                result = _snap_to_half(sol.x, [(n * m, R), (m * p, R), (p * n, R)])
                if result is not None:
                    return result
        
        return None

    # Known lower bounds:
    # For 2x3x9, the tensor rank is at most 54 (schoolbook).
    # Let's try to find lower ranks.
    # Strassen-like or other structured decompositions might help.
    # However, generic local search is what we have.
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 48 down to 40 (skipping 45-47 which might be too hard or known not to work)
    # Actually, let's try a range. The previous code tried 45-53.
    # Let's be more aggressive with restarts and time.
    
    # Prioritize lower ranks
    for R in range(42, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Increase restarts for lower ranks as they are harder
        n_restarts = 15 if R < 48 else 8
        result = _try_rank(R, n_restarts=n_restarts, method='lm')
        if result is not None:
            best_result = result
            best_rank = R
            break # Found a valid one, stop
    
    # If LM didn't find anything, try TRF which might handle non-convexity better
    if best_result is None:
        for R in range(45, best_rank):
            if time.time() > deadline - 10:
                break
            result = _try_rank(R, n_restarts=5, method='trf')
            if result is not None:
                best_result = result
                best_rank = R
                break

    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Increased restarts and tighter tolerances for LM/TFR optimization allow finding non-trivial low-rank decompositions (e.g., rank 45-50) that snap to half-integers, beating the schoolbook rank of 54.", q: 0.4, delta: [0.0, 0.05]}
