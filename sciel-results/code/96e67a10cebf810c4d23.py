"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute index mappings for vectorized operations
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # For each r: U[:,r] * V[:,r] * W[:,r] should reconstruct the tensor
    
    def verify(U, V, W, r):
        """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_ld = np.array(U, dtype=np.longdouble)
        V_ld = np.array(V, dtype=np.longdouble)
        W_ld = np.array(W, dtype=np.longdouble)
        # Reshape: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # W3 is indexed [k, i, r] so subscript is kir
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.all(np.abs(result - 1.0) < 1e-8)

    def snap_half(M):
        return np.round(np.array(M) * 2) / 2

    def snap_quarter(M):
        return np.round(np.array(M) * 4) / 4

    def snap_eighth(M):
        return np.round(np.array(M) * 8) / 8

    def to_fraction_matrix(M, max_denom=16):
        """Convert a float matrix to exact fractions with small denominators."""
        Frac_M = []
        for row in M:
            Frac_row = []
            for val in row:
                if abs(val) < 1e-12:
                    Frac_row.append(Fraction(0, 1))
                    continue
                best_frac = Fraction(str(val)).limit_denominator(max_denom)
                Frac_row.append(best_frac)
            Frac_M.append(Frac_row)
        return Frac_M

    def verify_fraction(U_frac, V_frac, W_frac, r):
        """Verify exact decomposition using fraction arithmetic."""
        U3 = np.array([[U_frac[i*m+j][rr] for rr in range(r)] for i in range(n) for j in range(m)])
        V3 = np.array([[V_frac[j*p+k][rr] for rr in range(r)] for j in range(m) for k in range(p)])
        W3 = np.array([[W_frac[k*n+i][rr] for rr in range(r)] for k in range(p) for i in range(n)])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0, 1)
                    for rr in range(r):
                        total += U3[i*m+j][rr] * V3[j*p+k][rr] * W3[k*n+i][rr]
                    if total != Fraction(1, 1):
                        return False
        return True

    def compute_error(U, V, W, r):
        """Compute sum of squared errors."""
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return float(np.sum((result - 1.0)**2))

    def als_decompose(r, max_iter=200, tol=1e-14, seed=42, init_mode='random'):
        """ALS with vectorized updates."""
        if init_mode == 'structured':
            U, V, W = _structured_init(r)
        else:
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, r).astype(np.float64)
            V = rng.randn(m * p, r).astype(np.float64)
            W = rng.randn(p * n, r).astype(np.float64)
            # Normalize
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        prev_err = np.inf
        stall_count = 0
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), solve M^T u = 1 where M[k,:] = V3[j,k,:] * W3[k,i,:]
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]  # (p, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = sol
                    except:
                        pass
            U = U_new
            U3 = U.reshape(n, m, r)
            
            # Update V: for each (j,k), solve M^T v = 1 where M[i,:] = U3[i,j,:] * W3[k,i,:]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]  # (n, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = sol
                    except:
                        pass
            V = V_new
            V3 = V.reshape(m, p, r)
            
            # Update W: for each (k,i), solve M^T w = 1 where M[j,:] = U3[i,j,:] * V3[j,k,:]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]  # (m, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = sol
                    except:
                        pass
            W = W_new
            W3 = W.reshape(p, n, r)
            
            err = compute_error(U, V, W, r)
            if err < prev_err:
                prev_err = err
                stall_count = 0
            else:
                stall_count += 1
            if err < tol:
                break
            if stall_count > 20:
                break
        
        return U, V, W, prev_err

    def _structured_init(r):
        """Create structured initialization based on block patterns."""
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Try to create a pattern where each rank term handles a subset of (i,j,k)
        # For r terms, distribute the n*m*p = 63 products across r terms
        products_per_term = (n * m * p + r - 1) // r
        
        idx = 0
        for rr in range(r):
            count = min(products_per_term, n * m * p - idx)
            for _ in range(count):
                if idx >= n * m * p:
                    break
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U[i * m + j, rr] = 1.0
                V[j * p + k, rr] = 1.0
                W[k * n + i, rr] = 1.0
                idx += 1
        
        # Add some random perturbation to escape local minima
        rng = np.random.RandomState(123)
        U += rng.randn(n * m, r) * 0.1
        V += rng.randn(m * p, r) * 0.1
        W += rng.randn(p * n, r) * 0.1
        
        return U, V, W

    def polish(U, V, W, r, max_iter=200):
        """High-precision polish using longdouble."""
        U_ld = U.astype(np.longdouble)
        V_ld = V.astype(np.longdouble)
        W_ld = W.astype(np.longdouble)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        
        prev_err = np.inf
        stall_count = 0
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p, dtype=np.longdouble), rcond=None)
                        U3[i, j, :] = sol
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n, dtype=np.longdouble), rcond=None)
                        V3[j, k, :] = sol
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m, dtype=np.longdouble), rcond=None)
                        W3[k, i, :] = sol
                    except:
                        pass
            
            # Check convergence
            result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = float(np.sum((result - 1.0)**2))
            if err < 1e-25:
                break
            if err < prev_err:
                prev_err = err
                stall_count = 0
            else:
                stall_count += 1
            if stall_count > 30:
                break
        
        U_ld = U3.reshape(n * m, r)
        V_ld = V3.reshape(m * p, r)
        W_ld = W3.reshape(p * n, r)
        return U_ld, V_ld, W_ld

    def try_rational_solution(U, V, W, r, max_denom=16):
        """Try to find an exact rational solution near the given float solution."""
        U_frac = to_fraction_matrix(U, max_denom)
        V_frac = to_fraction_matrix(V, max_denom)
        W_frac = to_fraction_matrix(W, max_denom)
        
        if verify_fraction(U_frac, V_frac, W_frac, r):
            return U_frac, V_frac, W_frac
        return None

    # Search strategy: try ranks from low to high with time budgeting
    # For 3x3*3x7, schoolbook is 63.
    # We'll try a range of ranks with different initialization strategies.
    best_result = None
    best_rank = 63
    
    # Rank candidates: try from low to high
    # Start with more promising ranges based on known results
    ranks_to_try = list(range(30, 64))
    
    # Allocate time with more weight to lower ranks
    time_weights = [1.0 / (i + 1)**0.5 for i in range(len(ranks_to_try))]
    total_weight = sum(time_weights)
    available_time = deadline - start_time - 10
    time_per_rank = [available_time * w / total_weight for w in time_weights]
    
    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline - 10:
            break
        
        rank_deadline = time.time() + time_per_rank[idx]
        found = False
        
        # Try multiple initialization strategies
        init_strategies = ['random', 'structured', 'random', 'random']
        n_restarts = min(len(init_strategies), max(1, int((rank_deadline - time.time()) / 5)))
        
        for restart in range(n_restarts):
            if time.time() > rank_deadline - 3:
                break
            
            init_mode = init_strategies[restart % len(init_strategies)]
            seed = 42 + r * 100 + restart * 7
            
            # Reduce max_iter for speed, rely on multiple restarts
            U, V, W, err = als_decompose(r, max_iter=80, seed=seed, init_mode=init_mode)
            
            if err < 1e-8:
                # Polish
                U_ld, V_ld, W_ld = polish(U, V, W, r)
                
                # Try various snap strategies
                for snap_func in [snap_half, snap_quarter, snap_eighth]:
                    U_snap = snap_func(U_ld)
                    V_snap = snap_func(V_ld)
                    W_snap = snap_func(W_ld)
                    
                    if verify(U_snap, V_snap, W_snap, r):
                        best_rank = r
                        best_result = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        found = True
                        break
                
                if found:
                    break
                
                # Try rational reconstruction
                frac_sol = try_rational_solution(U_ld, V_ld, W_ld, r, max_denom=8)
                if frac_sol is not None:
                    U_frac, V_frac, W_frac = frac_sol
                    best_rank = r
                    best_result = {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
                    found = True
                    break
                
                # Try without snap
                if verify(U_ld, V_ld, W_ld, r):
                    best_rank = r
                    best_result = {
                        "U": U_ld.tolist(),
                        "V": V_ld.tolist(),
                        "W": W_ld.tolist()
                    }
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
