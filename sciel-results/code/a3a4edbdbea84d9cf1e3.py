import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Target tensor T
    # T is of shape (n*m, m*p, n*p)
    # T[i*m+j, j*p+k, k*n+i] = 1 (Note: index mapping for W is k*n+i)
    # Wait, the task says W[k*n+i, r].
    # So W index is k*n + i.
    # U index is i*m + j.
    # V index is j*p + k.
    
    # Let's build the target tensor as a sparse representation or dense array.
    # Dense array size: 20 * 25 * 20 = 10000. This is small enough.
    
    T_flat = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u, idx_v, idx_w] = 1.0

    # Try different ranks
    # Known lower bound is around 25-30. Schoolbook is 100.
    # Strassen for 2x2x2 is 7.
    # For 4x4x4, rank is at most 49.
    # Let's try R = 40, 50.
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((n*p, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    for R in [40, 50, 60]:
        if time.time() > deadline - 20:
            break
            
        def func(params):
            U = params[:R*n*m].reshape(n*m, R)
            V = params[R*n*m:R*n*m + R*m*p].reshape(m*p, R)
            W = params[R*n*m + R*m*p:].reshape(n*p, R)
            
            # Compute S = sum_r u_r v_r w_r
            # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Use einsum for speed
            S = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Residual
            res = S - T_flat
            return res.ravel()
        
        # Initialize with random small values
        np.random.seed(42)
        x0 = np.random.randn(R * (n*m + m*p + n*p)) * 0.1
        
        try:
            # Use least squares
            result = least_squares(func, x0, method='lm', max_nfev=1000)
            
            if result.success or result.cost < 1e-6:
                # Verify exactness
                U_sol = result.x[:R*n*m].reshape(n*m, R)
                V_sol = result.x[R*n*m:R*n*m + R*m*p].reshape(m*p, R)
                W_sol = result.x[R*n*m + R*m*p:].reshape(n*p, R)
                
                S_sol = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                err = np.max(np.abs(S_sol - T_flat))
                
                if err < 1e-8:
                    # Round to integers if possible? The task allows integer/half-integer.
                    # But float is fine if it's exact? "integer/half-integer entries"
                    # We should try to snap to nearest 0.5
                    U_sol = np.round(U_sol * 2) / 2
                    V_sol = np.round(V_sol * 2) / 2
                    W_sol = np.round(W_sol * 2) / 2
                    
                    # Re-verify after rounding
                    S_round = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                    err_round = np.max(np.abs(S_round - T_flat))
                    
                    if err_round < 1e-8:
                        if R < best_rank:
                            best_rank = R
                            best_result = {
                                "U": U_sol.tolist(),
                                "V": V_sol.tolist(),
                                "W": W_sol.tolist()
                            }
        except Exception as e:
            pass

    return best_result
# EVOLVE-BLOCK-END
