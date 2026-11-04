import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find rank 9 decomposition for M(2,3,3)
    # Dimensions for sub-problem: n=2, m=3, p=3
    n_sub, m_sub, p_sub = 2, 3, 3
    rank_sub = 9
    
    # Target tensor for M(2,3,3)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Flatten indices:
    # U_idx: i*m + j (size 6)
    # V_idx: j*p + k (size 9)
    # W_idx: k*n + i (size 6)
    
    # We want U (6x9), V (9x9), W (6x9) such that sum_r U[:,r] V[:,r] W[:,r] == T
    # T is a 6x9x6 tensor.
    
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

    # Initialize U, V, W randomly
    np.random.seed(42)
    U_try = np.random.randn(n_sub * m_sub, rank_sub)
    V_try = np.random.randn(m_sub * p_sub, rank_sub)
    W_try = np.random.randn(p_sub * n_sub, rank_sub)
    
    # Optimization objective
    def objective(params):
        U_flat = params[:n_sub * m_sub * rank_sub]
        V_flat = params[n_sub * m_sub * rank_sub : n_sub * m_sub * rank_sub + m_sub * p_sub * rank_sub]
        W_flat = params[n_sub * m_sub * rank_sub + m_sub * p_sub * rank_sub :]
        
        U = U_flat.reshape(n_sub * m_sub, rank_sub)
        V = V_flat.reshape(m_sub * p_sub, rank_sub)
        W = W_flat.reshape(p_sub * n_sub, rank_sub)
        
        # Compute tensor
        # T_calc[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Using einsum
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Error
        return np.linalg.norm(T_calc - T_sub)

    # Use scipy if available
    try:
        from scipy.optimize import least_squares
        # Flatten initial guess
        x0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
        
        # Time limit check
        if time.time() - start_time > deadline - 10:
            raise TimeoutError("Time limit exceeded before optimization")
            
        # Run optimization
        # This might be slow, so we limit max_nfev
        res = least_squares(objective, x0, method='trf', max_nfev=500)
        
        if res.success or res.fun < 1e-6:
            # Extract solution
            U_sol = res.x[:n_sub * m_sub * rank_sub].reshape(n_sub * m_sub, rank_sub)
            V_sol = res.x[n_sub * m_sub * rank_sub : n_sub * m_sub * rank_sub + m_sub * p_sub * rank_sub].reshape(m_sub * p_sub, rank_sub)
            W_sol = res.x[n_sub * m_sub * rank_sub + m_sub * p_sub * rank_sub :].reshape(p_sub * n_sub, rank_sub)
            
            # Check error
            if res.fun < 1e-5:
                # Snap to nearest integers or half-integers?
                # The task allows integer/half-integer.
                # If the solution is close to rational, we can snap.
                # But for now, if it's exact enough, we can use it.
                # However, the evaluator requires "integer/half-integer entries".
                # If the numerical solution is not exactly rational, we might fail.
                # Let's try to snap to 0.5 increments.
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify snapped solution
                T_check = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.linalg.norm(T_check - T_sub) < 1e-5:
                    # Construct M(2,3,6) from two M(2,3,3) blocks
                    # U is same for both blocks
                    # V is split into V1 (for B cols 1-3) and V2 (for B cols 4-6)
                    # W is split into W1 (for C cols 1-3) and W2 (for C cols 4-6)
                    
                    U_final = np.zeros((n * m, 2 * rank_sub))
                    V_final = np.zeros((m * p, 2 * rank_sub))
                    W_final = np.zeros((p * n, 2 * rank_sub))
                    
                    # Block 1: cols 0-2 of B, cols 0-2 of C
                    U_final[:, :rank_sub] = U_snap
                    # V1 acts on B[:, 0:3]
                    # V_snap is 9x9. B1 is 3x3.
                    # V_final[:, :rank_sub] should correspond to B1.
                    # V_snap indices: j*p_sub + k. j in 0..2, k in 0..2.
                    # This matches B1 indices.
                    V_final[:, :rank_sub] = V_snap
                    # W1 acts on C[:, 0:3]
                    # W_snap indices: k*n_sub + i. k in 0..2.
                    # C1 indices: k*n + i. k in 0..2.
                    # Matches.
                    W_final[:, :rank_sub] = W_snap
                    
                    # Block 2: cols 3-5 of B, cols 3-5 of C
                    U_final[:, rank_sub:] = U_snap
                    # V2 acts on B[:, 3:6]
                    # V2 indices: j*p + (k+3).
                    # We need to shift the V_snap indices.
                    # V_snap has indices j*3 + k.
                    # We need indices j*6 + (k+3).
                    V2 = np.zeros((m * p, rank_sub))
                    for j in range(m_sub):
                        for k in range(p_sub):
                            # Map from (j, k) in V_snap to (j, k+3) in V2
                            V2[j * p + (k + 3), :] = V_snap[j * p_sub + k, :]
                    V_final[:, rank_sub:] = V2
                    
                    # W2 acts on C[:, 3:6]
                    # W2 indices: (k+3)*n + i.
                    W2 = np.zeros((p * n, rank_sub))
                    for k in range(p_sub):
                        for i in range(n_sub):
                            # Map from (k, i) in W_snap to (k+3, i) in W2
                            W2[(k + 3) * n + i, :] = W_snap[k * n_sub + i, :]
                    W_final[:, rank_sub:] = W2
                    
                    # Return
                    return {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
                
    except Exception as e:
        pass

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
