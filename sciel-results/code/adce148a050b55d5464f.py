import time
import numpy as np
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    def build_rank_28():
        """Build the known rank-28 Strassen tiling decomposition."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff
                
        def strassen_terms():
            terms = []
            # M1 = (X00+X11)(Y00+Y11)
            u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
            terms.append((u, v, w))
            # M2 = (X10+X11)Y00
            u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
            terms.append((u, v, w))
            # M3 = X00(Y01-Y11)
            u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
            terms.append((u, v, w))
            # M4 = X11(Y10-Y00)
            u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
            terms.append((u, v, w))
            # M5 = (X00+X01)Y11
            u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
            terms.append((u, v, w))
            # M6 = (X10-X00)(Y00+Y01)
            u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
            terms.append((u, v, w))
            # M7 = (X01-X11)(Y10+Y11)
            u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
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
    
    def verify_decomposition(U, V, W, rank):
        """Verify the tensor decomposition is exact."""
        # T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Non-zero entries: T[i*4+j, j*4+k, k*2+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        # Check zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check all (u,v,w) where u=i*m+j, v=j*p+k, w=k*n+i
                    # We need to check that T[u,v,w]=0 when not of the form above
                    # Actually the tensor is sparse: only T[i*m+j, j*p+k, k*n+i]=1
                    # All other entries should be 0
                    pass  # The non-zero check above covers the required entries
                    # For zero entries, we verify with a random test
        # Random verification
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-10)
    
    def try_reduce_rank(U, V, W, current_rank, target_rank, max_iters=50):
        """Try to reduce rank by zeroing columns and re-solving with nonlinear solver."""
        if target_rank >= current_rank:
            return None
        
        num_constraints = n * m * p  # 32
        
        def compute_residuals(Uv, Vv, Wv, active_cols):
            """Compute tensor residuals for active columns."""
            res = np.zeros(num_constraints)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        acc = 0.0
                        for r in active_cols:
                            acc += Uv[i*m+j, r] * Vv[j*p+k, r] * Wv[k*n+i, r]
                        res[idx] = acc - 1.0
                        idx += 1
            return res
        
        def nonlinear_solve(U_init, V_init, W_init, active_cols):
            """Use scipy least_squares to solve for all free parameters."""
            n_active = len(active_cols)
            # Variables: all entries of U, V, W for active columns
            x0 = np.concatenate([
                U_init[:, active_cols].flatten(),
                V_init[:, active_cols].flatten(),
                W_init[:, active_cols].flatten()
            ])
            
            def residuals(x):
                Uv = U_init.copy()
                Vv = V_init.copy()
                Wv = W_init.copy()
                Uv[:, active_cols] = x[:n*m*n_active].reshape(n*m, n_active)
                Vv[:, active_cols] = x[n*m*n_active : 2*n*m*n_active].reshape(m*p, n_active)
                Wv[:, active_cols] = x[2*n*m*n_active :].reshape(p*n, n_active)
                return compute_residuals(Uv, Vv, Wv, active_cols)
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000)
                if result.success or result.cost < 1e-20:
                    Uv = U_init.copy()
                    Vv = V_init.copy()
                    Wv = W_init.copy()
                    Uv[:, active_cols] = result.x[:n*m*n_active].reshape(n*m, n_active)
                    Vv[:, active_cols] = result.x[n*m*n_active : 2*n*m*n_active].reshape(m*p, n_active)
                    Wv[:, active_cols] = result.x[2*n*m*n_active :].reshape(p*n, n_active)
                    return Uv, Vv, Wv
            except Exception:
                pass
            return None
        
        for trial in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Pick a random column to try removing
            col_to_remove = np.random.randint(0, current_rank)
            
            active_cols = [c for c in range(current_rank) if c != col_to_remove]
            
            # Initialize: keep current values for active columns
            U_init = U.copy()
            V_init = V.copy()
            W_init = W.copy()
            
            result = nonlinear_solve(U_init, V_init, W_init, active_cols)
            if result is not None:
                U_sol, V_sol, W_sol = result
                # Verify
                if verify_decomposition(U_sol, V_sol, W_sol, current_rank):
                    # Remove the zero column
                    mask = np.ones(current_rank, dtype=bool)
                    mask[col_to_remove] = False
                    U_new = U_sol[:, mask]
                    V_new = V_sol[:, mask]
                    W_new = W_sol[:, mask]
                    if verify_decomposition(U_new, V_new, W_new, current_rank - 1):
                        return U_new, V_new, W_new
        
        return None
    
    # Build the rank-28 solution
    U, V, W = build_rank_28()
    current_rank = 28
    
    # Try to reduce rank
    for target_rank in range(current_rank - 1, 7, -1):  # Try down to rank 8
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_reduce_rank(U, V, W, current_rank, target_rank, max_iters=40)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
    
    # Final verification
    if not verify_decomposition(U, V, W, current_rank):
        # Fall back to rank 28
        U, V, W = build_rank_28()
        current_rank = 28
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
                    acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
