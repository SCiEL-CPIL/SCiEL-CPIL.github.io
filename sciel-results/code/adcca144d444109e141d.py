import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 5
    DEADLINE = 145.0
    start_time = time.time()
    
    # Precompute target tensor T of shape (4, 10, 10)
    # U index: i*m + j  (0..3)
    # V index: j*p + k  (0..9)
    # W index: k*n + i  (0..9)
    # T[u, v, w] = 1 if u=(i,j), v=(j',k), w=(k',i') and i==i', j==j', k==k'
    T = np.zeros((4, 10, 10), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def error(params, r):
        if time.time() > DEADLINE - 5:
            return 1e10
        U = params[:4*r].reshape((4, r))
        V = params[4*r:14*r].reshape((10, r))
        W = params[14*r:].reshape((10, r))
        
        # Compute S[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
        # Using einsum: 'ut, vt, wt -> uvw'
        S = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        err = np.sum((S - T)**2)
        return err

    def check_exact(U, V, W, r, tol=1e-5):
        S = np.einsum('ut,vt,wt->uvw', U, V, W)
        return np.allclose(S, T, atol=tol)

    def snap_to_half_integers(arr, tol=1e-5):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_find_decomposition(r):
        num_vars = 24 * r
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10:
                return None
            
            # Random initialization
            # Try different scales and sparsity
            if restart % 4 == 0:
                np.random.seed(restart * 7 + r)
                x0 = np.random.uniform(-1, 1, size=num_vars)
            elif restart % 4 == 1:
                np.random.seed(restart * 13 + r)
                x0 = np.random.uniform(-2, 2, size=num_vars)
            elif restart % 4 == 2:
                # Sparse init
                np.random.seed(restart * 17 + r)
                x0 = np.zeros(num_vars)
                num_non_zero = num_vars // 3
                indices = np.random.choice(num_vars, size=num_non_zero, replace=False)
                x0[indices] = np.random.choice([-1, 1], size=num_non_zero)
            else:
                # Integer-ish init
                np.random.seed(restart * 19 + r)
                x0 = np.random.choice([-1, 0, 1], size=num_vars)
            
            # Optimize
            # Use L-BFGS-B for speed and robustness
            res = minimize(error, x0, args=(r,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < 1e-8:
                U = res.x[:4*r].reshape((4, r))
                V = res.x[4*r:14*r].reshape((10, r))
                W = res.x[14*r:].reshape((10, r))
                
                # Snap to half-integers
                U_r = snap_to_half_integers(U)
                V_r = snap_to_half_integers(V)
                W_r = snap_to_half_integers(W)
                
                if U_r is not None and V_r is not None and W_r is not None:
                    if check_exact(U_r, V_r, W_r, r, tol=1e-6):
                        return (U_r, V_r, W_r)
                
                # If raw float is exact, try harder snapping
                if check_exact(U, V, W, r, tol=1e-6):
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if check_exact(U_i, V_i, W_i, r, tol=1e-6):
                        return (U_i, V_i, W_i)
                    
                    U_r2 = np.round(U * 2) / 2
                    V_r2 = np.round(V * 2) / 2
                    W_r2 = np.round(W * 2) / 2
                    if check_exact(U_r2, V_r2, W_r2, r, tol=1e-6):
                        return (U_r2, V_r2, W_r2)

        return None

    # Try rank 17
    decomp = try_find_decomposition(17)
    if decomp:
        U, V, W = decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 16
    if time.time() < DEADLINE - 30:
        decomp = try_find_decomposition(16)
        if decomp:
            U, V, W = decomp
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 18
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(10)]
        W = [[0] * rank for _ in range(10)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*5 + k0][t_offset+0] = 1
            V[1*5 + k1][t_offset+0] = 1
            V[0*5 + k0][t_offset+1] = 1
            V[0*5 + k1][t_offset+2] = 1
            V[1*5 + k1][t_offset+2] = -1
            V[1*5 + k0][t_offset+3] = 1
            V[0*5 + k0][t_offset+3] = -1
            V[1*5 + k1][t_offset+4] = 1
            V[0*5 + k0][t_offset+5] = 1
            V[0*5 + k1][t_offset+5] = 1
            V[1*5 + k0][t_offset+6] = 1
            V[1*5 + k1][t_offset+6] = 1
            
            W[k0*2 + 0][t_offset+0] = 1
            W[k1*2 + 1][t_offset+0] = 1
            W[k0*2 + 1][t_offset+1] = 1
            W[k1*2 + 1][t_offset+1] = -1
            W[k1*2 + 0][t_offset+2] = 1
            W[k1*2 + 1][t_offset+2] = 1
            W[k0*2 + 0][t_offset+3] = 1
            W[k0*2 + 1][t_offset+3] = 1
            W[k0*2 + 0][t_offset+4] = -1
            W[k1*2 + 0][t_offset+4] = 1
            W[k1*2 + 1][t_offset+5] = 1
            W[k0*2 + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*5 + 4][t] = 1
        W[4*2 + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*5 + 4][t] = 1
        W[4*2 + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*5 + 4][t] = 1
        W[4*2 + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*5 + 4][t] = 1
        W[4*2 + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    return get_rank_18_solution()
