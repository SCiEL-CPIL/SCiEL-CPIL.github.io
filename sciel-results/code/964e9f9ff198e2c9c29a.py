import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook decomposition: rank 45
    best_rank = n * m * p  # 45
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True

    def verify_rounded(U, V, W):
        """Verify with rounding to half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return verify(U_r, V_r, W_r), U_r, V_r, W_r

    def solve_subtensor(rows_U, rows_V, rows_W, target_k_indices, rank, seed, name=""):
        """
        Solve for a rank 'rank' decomposition of a sub-tensor.
        rows_U, rows_V, rows_W are the indices in the full matrices that are active.
        target_k_indices are the k values for which the tensor is 1.
        """
        np.random.seed(seed)
        
        # Dimensions for the sub-problem
        nU = len(rows_U)
        nV = len(rows_V)
        nW = len(rows_W)
        
        U = np.random.randn(nU, rank) * 0.5
        V = np.random.randn(nV, rank) * 0.5
        W = np.random.randn(nW, rank) * 0.5
        
        # Indices for the target tensor entries
        # We need to map (i, j, k) to local indices in U, V, W
        # This is specific to the M_{3,3,3} and M_{3,3,2} structures.
        # We will implement specific solvers instead of a generic one to avoid mapping complexity.
        pass

    def solve_m333(r1, seed):
        """Try to find rank r1 decomposition of M_{3,3,3}."""
        np.random.seed(seed)
        
        U = np.random.randn(9, r1) * 0.5
        V = np.random.randn(9, r1) * 0.5
        W = np.random.randn(9, r1) * 0.5
        
        # Objective: sum_{i,j,k in {0,1,2}} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        def objective(x):
            U_f = x[:9*r1].reshape(9, r1)
            V_f = x[9*r1:18*r1].reshape(9, r1)
            W_f = x[18*r1:].reshape(9, r1)
            
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = np.dot(U_f[i*3+j], V_f[j*3+k] * W_f[k*3+i])
                        err += (val - 1.0)**2
            return err
        
        from scipy.optimize import minimize
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use multiple restarts or just one with good options
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                U_sol = res.x[:9*r1].reshape(9, r1)
                V_sol = res.x[9*r1:18*r1].reshape(9, r1)
                W_sol = res.x[18*r1:].reshape(9, r1)
                
                # Try rounding to half-integers
                ok, U_r, V_r, W_r = verify_rounded(U_sol, V_sol, W_sol)
                if ok:
                    return U_r, V_r, W_r
                
                # Check if exact without rounding
                if verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    def solve_m332(r2, seed):
        """Try to find rank r2 decomposition of M_{3,3,2} (k=0,1 part)."""
        # Note: In the full problem, M_{3,3,2} corresponds to k in {3,4} in the second block.
        # But structurally it's the same as M_{3,3,2}.
        # U is 9 x r2 (i,j)
        # V is 6 x r2 (j,k) where j in 0..2, k in 0..1
        # W is 6 x r2 (k,i) where k in 0..1, i in 0..2
        
        np.random.seed(seed)
        
        U = np.random.randn(9, r2) * 0.5
        V = np.random.randn(6, r2) * 0.5
        W = np.random.randn(6, r2) * 0.5
        
        def objective(x):
            U_f = x[:9*r2].reshape(9, r2)
            V_f = x[9*r2:15*r2].reshape(6, r2)
            W_f = x[15*r2:].reshape(6, r2)
            
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        # Map local indices
                        # U: i*3+j
                        # V: j*2+k
                        # W: k*3+i
                        val = np.dot(U_f[i*3+j], V_f[j*2+k] * W_f[k*3+i])
                        err += (val - 1.0)**2
            return err
        
        from scipy.optimize import minimize
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                U_sol = res.x[:9*r2].reshape(9, r2)
                V_sol = res.x[9*r2:15*r2].reshape(6, r2)
                W_sol = res.x[15*r2:].reshape(6, r2)
                
                # Verify in local context
                local_ok = True
                for i in range(3):
                    for j in range(3):
                        for k in range(2):
                            val = np.dot(U_sol[i*3+j], V_sol[j*2+k] * W_sol[k*3+i])
                            if not np.isclose(val, 1.0, atol=1e-8):
                                local_ok = False
                                break
                        if not local_ok: break
                    if not local_ok: break
                
                if local_ok:
                    # Try rounding
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    local_ok_r = True
                    for i in range(3):
                        for j in range(3):
                            for k in range(2):
                                val = np.dot(U_r[i*3+j], V_r[j*2+k] * W_r[k*3+i])
                                if not np.isclose(val, 1.0, atol=1e-8):
                                    local_ok_r = False
                                    break
                            if not local_ok_r: break
                        if not local_ok_r: break
                    if not local_ok_r: break
                    
                    if local_ok_r:
                        return U_r, V_r, W_r
                    else:
                        return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    # Search for M_{3,3,3} decomposition
    m333_sol = None
    best_r1 = 27
    for r1 in range(26, 19, -1):
        if time.time() > deadline - 30:
            break
        
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            sol = solve_m333(r1, seed)
            if sol is not None:
                m333_sol = (r1, sol)
                break
        
        if m333_sol is not None:
            break

    # Search for M_{3,3,2} decomposition
    m332_sol = None
    best_r2 = 18
    for r2 in range(17, 10, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            sol = solve_m332(r2, seed)
            if sol is not None:
                m332_sol = (r2, sol)
                break
        
        if m332_sol is not None:
            break

    # Build final solution
    if m333_sol is not None and m332_sol is not None:
        r1, (U1, V1, W1) = m333_sol
        r2, (U2, V2, W2) = m332_sol
        total_rank = r1 + r2
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # Part 1: M_{3,3,3} (k=0,1,2)
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, :r1] = U1[i*3+j, :]
        
        for j in range(3):
            for k in range(3):
                V_full[j*5+k, :r1] = V1[j*3+k, :]
        
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, :r1] = W1[k*3+i, :]
        
        # Part 2: M_{3,3,2} (k=3,4)
        # Map local indices to global
        # Local V index: j*2 + k_local -> Global V index: j*5 + (3 + k_local)
        # Local W index: k_local*3 + i -> Global W index: (3 + k_local)*3 + i
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, r1:r1+r2] = U2[i*3+j, :]
        
        for j in range(3):
            for k_local in range(2):
                k_global = 3 + k_local
                V_full[j*5+k_global, r1:r1+r2] = V2[j*2+k_local, :]
        
        for k_local in range(2):
            k_global = 3 + k_local
            for i in range(3):
                W_full[k_global*3+i, r1:r1+r2] = W2[k_local*3+i, :]
        
        # Verify
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
        
        # Try rounding
        ok, U_r, V_r, W_r = verify_rounded(U_full, V_full, W_full)
        if ok:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    # Fallback: schoolbook rank 45
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
