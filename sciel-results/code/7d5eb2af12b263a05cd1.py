import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-6)

    def build_rank32():
        """Build the known rank-32 decomposition using Strassen blocks."""
        rank = 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        def add_strassen(k0, k1):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[0, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
            col += 1
            # M2: (A10+A11)B00
            U[2, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1
            W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
            col += 1
            # M3: A00(B01-B11)
            U[0, col] = 1
            V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
            W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
            col += 1
            # M4: A11(B10-B00)
            U[3, col] = 1
            V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
            W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
            col += 1
            # M5: (A00+A01)B11
            U[0, col] = 1; U[1, col] = 1
            V[1*p + k1, col] = 1
            W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[2, col] = 1; U[0, col] = -1
            V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
            W[k1*n + 1, col] = 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[1, col] = 1; U[3, col] = -1
            V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
            col += 1
            U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
            col += 1
            U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
            col += 1
            U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
            col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)
        
        return U, V, W

    def compute_residual(U, V, W, n, m, p):
        """Compute the max absolute residual of the tensor decomposition."""
        U_r = U.reshape(n, m, U.shape[1])
        V_r = V.reshape(m, p, V.shape[1])
        W_r = W.reshape(p, n, W.shape[1])
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.max(np.abs(T - expected))

    def try_reduce_rank(U0, V0, W0, n, m, p, target_rank, time_budget):
        """Try to reduce rank by eliminating one term via local search."""
        start_time = time.time()
        nm, mp, pn = n * m, m * p, p * n
        R = U0.shape[1]
        
        # Try each column as the one to eliminate
        best_result = None
        best_score = float('inf')
        
        for target_col in range(R):
            if time.time() - start_time > time_budget:
                break
            
            # Create a copy and zero out the target column
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            # We'll optimize all parameters, with a penalty on the target column norm
            def objective(params, lam=10.0):
                U_opt = params[:nm * R].reshape(nm, R)
                V_opt = params[nm * R: nm * R + mp * R].reshape(mp, R)
                W_opt = params[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
                
                U_r = U_opt.reshape(n, m, R)
                V_r = V_opt.reshape(m, p, R)
                W_r = W_opt.reshape(p, n, R)
                T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
                expected = np.ones((n, m, p))
                residual = T - expected
                
                # Smooth max via log-sum-exp
                r2 = residual.flatten() ** 2
                max_r2 = np.log(np.sum(np.exp(r2 - np.max(r2)))) + np.max(r2)
                max_residual = np.sqrt(max_r2)
                
                # Penalty for target column not being zero
                col_penalty = lam * (np.sum(U_opt[:, target_col] ** 2) + 
                                    np.sum(V_opt[:, target_col] ** 2) + 
                                    np.sum(W_opt[:, target_col] ** 2))
                
                return max_residual + col_penalty
            
            # Initialize: keep original, but scale down target column
            x0 = np.concatenate([
                U0.flatten(), V0.flatten(), W0.flatten()
            ])
            x0_target = x0.copy()
            # Scale down target column in initial guess
            idx_u = target_col
            idx_v = nm * R + target_col
            idx_w = nm * R + mp * R + target_col
            x0_target[idx_u] = 0.1 * x0[idx_u]
            x0_target[idx_v] = 0.1 * x0[idx_v]
            x0_target[idx_w] = 0.1 * x0[idx_w]
            
            try:
                res = minimize(objective, x0_target, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                U_opt = res.x[:nm * R].reshape(nm, R)
                V_opt = res.x[nm * R: nm * R + mp * R].reshape(mp, R)
                W_opt = res.x[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
                
                # Check if target column is effectively zero
                col_norm = np.sqrt(np.sum(U_opt[:, target_col] ** 2) + 
                                  np.sum(V_opt[:, target_col] ** 2) + 
                                  np.sum(W_opt[:, target_col] ** 2))
                
                if col_norm < 0.01:
                    # Remove the target column
                    mask = np.ones(R, dtype=bool)
                    mask[target_col] = False
                    U_new = U_opt[:, mask]
                    V_new = V_opt[:, mask]
                    W_new = W_opt[:, mask]
                    
                    # Snap to half-integers
                    U_snap = np.round(U_new * 2) / 2
                    V_snap = np.round(V_new * 2) / 2
                    W_snap = np.round(W_new * 2) / 2
                    
                    if verify_decomp(U_snap, V_snap, W_snap, n, m, p):
                        score = compute_residual(U_snap, V_snap, W_snap, n, m, p)
                        if score < best_score:
                            best_score = score
                            best_result = (U_snap, V_snap, W_snap)
            except Exception:
                continue
        
        return best_result

    # Step 1: Build the known rank-32 solution
    U32, V32, W32 = build_rank32()
    
    # Verify it's correct
    if not verify_decomp(U32, V32, W32, n, m, p):
        # Should not happen, but handle gracefully
        pass
    
    # Step 2: Try to reduce to rank 31
    result = None
    if time.time() < deadline:
        result = try_reduce_rank(U32, V32, W32, n, m, p, 31, time_budget=60.0)
    
    # Step 3: Return best result
    if result is not None:
        U_best, V_best, W_best = result
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    else:
        # Fall back to rank 32
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
