import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def _get_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a,b,c] where a=i*m+j, b=jp*p+k, c=kp*n+ip."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):
                for k in range(p):
                    b = jp*p + k
                    for ip in range(n):
                        for kp in range(p):
                            c = kp*n + ip
                            if i == ip and j == jp and k == kp:
                                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    T_target = _get_target_tensor(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T_comp - T_target))
    return max_err < tol, max_err

def _build_rank14_strassen(n, m, p):
    """Build the known rank-14 Strassen decomposition for 2x2 * 2x4."""
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx, col] = val
        for idx, val in v_dict.items():
            V[idx, col] = val
        for idx, val in w_dict.items():
            W[idx, col] = val
            
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    # Left block (B[:, 0:2])
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Right block (B[:, 2:4])
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return U, V, W

def _try_rank13_peel(n, m, p, U14, V14, W14, deadline):
    """
    Try to find a rank-13 decomposition by 'peeling' the 14th term.
    Strategy: fix one factor of the 14th term to a canonical unit vector,
    then solve for the remaining 13 terms.
    
    For each choice of which factor to fix (U, V, or W) and which row to fix,
    we set that factor's 14th column to a unit vector and solve for the rest.
    """
    T_target = _get_target_tensor(n, m, p)
    n_m = n * m  # 4
    m_p = m * p  # 8
    p_n = p * n  # 8
    
    # The 14th term (index 13) in the rank-14 decomposition
    # We'll try to absorb it into the first 13 terms by adjusting them.
    # 
    # Approach: For each canonical unit vector e_a in the fixed factor's space,
    # we set the 14th column of that factor to e_a, and then need to find
    # 13 vectors in each of the other two factors such that:
    # sum_{r=0}^{12} U[a,r]V[b,r]W[c,r] + U[a,13]V[b,13]W[c,13] = T[a,b,c]
    # where one of the 14th-column vectors is fixed to a unit vector.
    #
    # This is still nonlinear. Instead, let's use a linear approach:
    # Fix the 14th column of, say, W to be e_c (unit vector at position c).
    # Then the constraint becomes:
    #   sum_{r=0}^{12} U[a,r]V[b,r]W[c,r] + V[b,13] * (1 if c'==c else 0) = T[a,b,c]
    # This is still bilinear in U and V.
    #
    # Better approach: Use the fact that we have 256 equations and 240 unknowns
    # for rank 12, but for rank 13 we have 4*13+8*13+8*13 = 234 unknowns and 256 equations.
    # So rank 13 is overdetermined unless there's special structure.
    #
    # Let's try a different strategy: directly search for rank 12 or 13 using
    # the known rank-14 as initialization and nonlinear optimization.
    
    # Try rank 13 with least_squares, initialized near the rank-14 solution
    for rank_target in [13, 12]:
        if time.time() > deadline:
            break
        
        n_vars = n_m * rank_target + m_p * rank_target + p_n * rank_target
        # For rank 13: 4*13 + 8*13 + 8*13 = 52 + 104 + 104 = 260
        # For rank 12: 4*12 + 8*12 + 8*12 = 48 + 96 + 96 = 240
        
        def residuals(x):
            U = x[:n_m*rank_target].reshape(n_m, rank_target)
            V = x[n_m*rank_target:n_m*rank_target+m_p*rank_target].reshape(m_p, rank_target)
            W = x[n_m*rank_target+m_p*rank_target:].reshape(p_n, rank_target)
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_comp - T_target).flatten()
        
        best_err = float('inf')
        best_x = None
        
        # Initialize near the rank-14 solution (pad/truncate to target rank)
        # Take first rank_target columns of the rank-14 solution
        if rank_target <= 14:
            U_init = U14[:, :rank_target]
            V_init = V14[:, :rank_target]
            W_init = W14[:, :rank_target]
        else:
            U_init = np.random.randn(n_m, rank_target) * 0.1
            V_init = np.random.randn(m_p, rank_target) * 0.1
            W_init = np.random.randn(p_n, rank_target) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few restarts
        n_restarts = 10
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                x_init = x0
            else:
                # Perturb the initialization
                scale = 0.5 ** (r // 3)
                x_init = x0 + np.random.randn(n_vars) * scale * 0.1
            
            try:
                sol = least_squares(residuals, x_init, method='lm', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                err = sol.cost
                if err < best_err:
                    best_err = err
                    best_x = sol.x
            except Exception:
                pass
        
        if best_x is not None and best_err < 1e-10:
            U = best_x[:n_m*rank_target].reshape(n_m, rank_target)
            V = best_x[n_m*rank_target:n_m*rank_target+m_p*rank_target].reshape(m_p, rank_target)
            W = best_x[n_m*rank_target+m_p*rank_target:].reshape(p_n, rank_target)
            
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            valid, max_err = _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p)
            if valid:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # If rounding doesn't work, try rounding to integers
            U_rounded_int = np.round(U)
            V_rounded_int = np.round(V)
            W_rounded_int = np.round(W)
            valid, max_err = _verify_decomposition(U_rounded_int, V_rounded_int, W_rounded_int, n, m, p)
            if valid:
                return {"U": U_rounded_int.tolist(), "V": V_rounded_int.tolist(), "W": W_rounded_int.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # First, try to find a rank-13 or rank-12 solution using the rank-14 as initialization
    U14, V14, W14 = _build_rank14_strassen(n, m, p)
    
    result = _try_rank13_peel(n, m, p, U14, V14, W14, deadline)
    if result is not None:
        return result
    
    # Fallback to rank 14
    return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
