import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Build the target tensor for 2x3x6
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # Schoolbook: rank = n*m*p = 36
    rank_school = n * m * p
    U_school = np.zeros((num_A, rank_school))
    V_school = np.zeros((num_B, rank_school))
    W_school = np.zeros((num_C, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try to find better decomposition using block-doubling
    # Solve 2x3x3 subproblem, then double to 2x3x6
    
    n2, m2, p2 = 2, 3, 3
    T_233 = np.zeros((n2, m2, p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_233[i, j, k] = 1
                
    T_flat = T_233.flatten()
    
    def verify_233(U, V, W, tol=1e-8):
        """Verify that U,V,W correctly decompose the 2x3x3 tensor."""
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T_comp.flatten() - T_flat
        return np.linalg.norm(diff) < tol
    
    def snap_to_half_integers(U, V, W, tol=1e-6):
        """Try to snap values to half-integers and verify."""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        T_comp = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
        diff = T_comp.flatten() - T_flat
        if np.linalg.norm(diff) < tol:
            return U_snap, V_snap, W_snap
        return U, V, W
    
    def solve_233(rank, n_restarts=20, max_nfev=2000):
        """Solve the 2x3x3 tensor decomposition problem for given rank."""
        if time.time() > deadline - 5:
            return None
        
        from scipy.optimize import least_squares
        
        def objective(params):
            U = params[0:n2*rank].reshape(n2, rank)
            V = params[n2*rank:n2*rank+m2*rank].reshape(m2, rank)
            W = params[n2*rank+m2*rank:].reshape(p2, rank)
            
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            return T_comp.flatten() - T_flat
        
        best_params = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Small random
                np.random.seed(42 + restart)
                params0 = np.random.randn((n2 + m2 + p2) * rank) * 0.1
            elif restart == 1:
                # Larger random
                np.random.seed(100 + restart)
                params0 = np.random.randn((n2 + m2 + p2) * rank) * 0.5
            elif restart == 2:
                # Structured initialization: try to match schoolbook pattern
                np.random.seed(200 + restart)
                params0 = np.random.randn((n2 + m2 + p2) * rank) * 0.3
            else:
                # Random with varying scale
                np.random.seed(300 + restart * 7)
                scale = 0.1 + 0.1 * (restart % 5)
                params0 = np.random.randn((n2 + m2 + p2) * rank) * scale
            
            try:
                res = least_squares(objective, params0, method='lm', max_nfev=max_nfev)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
            except Exception:
                pass
        
        if best_params is None or best_err > 1e-8:
            return None
        
        U = best_params[0:n2*rank].reshape(n2, rank)
        V = best_params[n2*rank:n2*rank+m2*rank].reshape(m2, rank)
        W = best_params[n2*rank+m2*rank:].reshape(p2, rank)
        
        # Try snapping to half-integers
        U, V, W = snap_to_half_integers(U, V, W)
        
        # Final verification
        if verify_233(U, V, W):
            return U, V, W
        
        return None
    
    def build_236_from_233(U_233, V_233, W_233, rank_233):
        """Build 2x3x6 decomposition from 2x3x3 solution using block-doubling."""
        rank_236 = 2 * rank_233
        U_236 = np.zeros((n*m, rank_236))
        V_236 = np.zeros((m*p, rank_236))
        W_236 = np.zeros((p*n, rank_236))
        
        # U is the same for both blocks (since n=2, m=3)
        U_236[:, 0:rank_233] = U_233
        U_236[:, rank_233:2*rank_233] = U_233
        
        # V: Block 0 uses B[0:3] (first 3 columns), Block 1 uses B[3:6]
        for j in range(m2):
            for k in range(p2):
                V_236[j*6 + k, 0:rank_233] = V_233[j, 0:rank_233]
                V_236[j*6 + (k+3), rank_233:2*rank_233] = V_233[j, 0:rank_233]
                
        # W: Block 0 uses C[0:3], Block 1 uses C[3:6]
        for k in range(p2):
            for i in range(n2):
                W_236[k*2 + i, 0:rank_233] = W_233[k, 0:rank_233]
                W_236[(k+3)*2 + i, rank_233:2*rank_233] = W_233[k, 0:rank_233]
        
        return U_236, V_236, W_236
    
    # Try to find 2x3x3 decomposition at various ranks
    # Theoretical lower bound for 2x3x3 is at least 8 (trivial) 
    # Known results suggest rank 12 might be achievable
    
    for target_rank in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 10:
            break
            
        result = solve_233(target_rank)
        if result is not None:
            U_233, V_233, W_233 = result
            U_236, V_236, W_236 = build_236_from_233(U_233, V_233, W_233, target_rank)
            
            # Verify the full 2x3x6 decomposition
            # Check a few random positions
            U_arr = U_236
            V_arr = V_236
            W_arr = W_236
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
    
    # If all fail, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
