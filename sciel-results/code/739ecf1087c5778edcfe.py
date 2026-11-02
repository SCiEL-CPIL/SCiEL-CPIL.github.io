"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
from scipy.optimize import minimize
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_start_time = time.time()
DEADLINE = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    # 2 Strassen blocks (rank 7 each) for first 4 cols, 1 Schoolbook (rank 4) for last col
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
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
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    # Attempt to find better decomposition for 2x2x3 part
    # If we find rank r for 2x2x3, total rank is 7 + r
    # Target: r < 11 to beat rank 18
    
    n_sub = 2
    m_sub = 2
    p_sub = 3
    
    def error_223(params, rank):
        U = params[:n_sub*m_sub*rank].reshape((n_sub*m_sub, rank))
        V = params[n_sub*m_sub*rank : n_sub*m_sub*rank + m_sub*p_sub*rank].reshape((m_sub*p_sub, rank))
        W = params[n_sub*m_sub*rank + m_sub*p_sub*rank :].reshape((p_sub*n_sub, rank))
        
        err = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        err += (val - target) ** 2
        return err

    def check_exact_223(U, V, W, rank, tol=1e-5):
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def als_step_U(V, W, rank, lam=0.01):
        U = np.zeros((n_sub*m_sub, rank))
        for i in range(n_sub*m_sub):
            # U[i, c] = sum_{jp, k} V[jp, k] * W[k, i] * U[i, c] (cyclic dependency)
            # Actually: U[i, c] * (sum_{jp, k} V[jp, k] * W[k, i] * U[i, c]) ... no.
            # The term in error is U[i, c] * V[jp, k, c] * W[k, i, c].
            # For fixed i, the contribution to error is sum_{jp, k, c} U[i, c] V[jp, k, c] W[k, i, c].
            # We want to minimize sum_{jp, k} (sum_c U[i, c] V[jp, k, c] W[k, i, c] - delta_{j, jp})^2.
            # Let y_jp = sum_c U[i, c] V[jp, k, c] W[k, i, c].
            # This is a quadratic form in U[i, :].
            # Hessian H[c, c'] = sum_{jp, k} V[jp, k, c] W[k, i, c] V[jp, k, c'] W[k, i, c']
            # Gradient G[c] = sum_{jp, k} (sum_{c'} U[i, c'] V[jp, k, c'] W[k, i, c'] - delta_{j, jp}) V[jp, k, c] W[k, i, c]
            # Note: delta_{j, jp} depends on j (index of i) and jp.
            # i = j * m_sub + j_idx. So j = i // m_sub, j_idx = i % m_sub.
            
            j_idx = i % m_sub
            # Build H and G
            H = np.zeros((rank, rank))
            G = np.zeros(rank)
            
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j_idx == jp else 0.0
                    v_vec = V[jp*p_sub + k, :]
                    w_val = W[k*n_sub + i//m_sub, :] # W index: k*n + i, here i is global row index for A, but in sub-problem it's local i.
                    # Wait, W index in sub problem is W[k_sub * n_sub + i_local].
                    # In the full problem, W index is W[k_global * n + i_global].
                    # Here we are solving for the sub-problem (cols 2,3,4).
                    # The indices are local.
                    w_vec = W[k*n_sub + i, :] # This is wrong. W is (p_sub*n_sub, rank). Index is k*n_sub + i.
                    # But i is the row index of U (0..3). W's second dimension is n_sub (0..1).
                    # So W[k*n_sub + i_local] where i_local is 0 or 1.
                    # In our loop i is 0..3. i_local = i % n_sub? No, i is the row of U.
                    # U is (n_sub*m_sub, rank). Row i corresponds to A[i//m_sub, i%m_sub].
                    # W is (p_sub*n_sub, rank). Row k*n_sub + i_local corresponds to C[k, i_local].
                    # The term is U[i, c] V[jp, k, c] W[k, i_local, c].
                    # So i_local = i // m_sub.
                    i_local = i // m_sub
                    w_vec = W[k*n_sub + i_local, :]
                    
                    # H += outer(v_vec * w_vec, v_vec * w_vec)
                    v_w = v_vec * w_vec
                    H += np.outer(v_w, v_w)
                    # G += (sum_c' U[i, c'] v_w[c'] - target) * v_w
                    current_sum = np.dot(U[i, :], v_w)
                    G += (current_sum - target) * v_w
            
            H += lam * np.eye(rank)
            try:
                U[i, :] = np.linalg.solve(H, -G)
            except:
                pass
        return U

    def als_step_V(U, W, rank, lam=0.01):
        V = np.zeros((m_sub*p_sub, rank))
        for jp in range(m_sub):
            for k in range(p_sub):
                idx = jp*p_sub + k
                H = np.zeros((rank, rank))
                G = np.zeros(rank)
                for i in range(n_sub*m_sub):
                    i_local = i // m_sub
                    j_idx = i % m_sub
                    target = 1.0 if j_idx == jp else 0.0
                    u_vec = U[i, :]
                    w_vec = W[k*n_sub + i_local, :]
                    u_w = u_vec * w_vec
                    H += np.outer(u_w, u_w)
                    current_sum = np.dot(V[idx, :], u_w)
                    G += (current_sum - target) * u_w
                H += lam * np.eye(rank)
                try:
                    V[idx, :] = np.linalg.solve(H, -G)
                except:
                    pass
        return V

    def als_step_W(U, V, rank, lam=0.01):
        W = np.zeros((p_sub*n_sub, rank))
        for k in range(p_sub):
            for i_local in range(n_sub):
                idx = k*n_sub + i_local
                H = np.zeros((rank, rank))
                G = np.zeros(rank)
                for i in range(n_sub*m_sub):
                    if i // m_sub != i_local:
                        continue
                    j_idx = i % m_sub
                    for jp in range(m_sub):
                        target = 1.0 if j_idx == jp else 0.0
                        u_vec = U[i, :]
                        v_vec = V[jp*p_sub + k, :]
                        u_v = u_vec * v_vec
                        H += np.outer(u_v, u_v)
                        current_sum = np.dot(W[idx, :], u_v)
                        G += (current_sum - target) * u_v
                H += lam * np.eye(rank)
                try:
                    W[idx, :] = np.linalg.solve(H, -G)
                except:
                    pass
        return W

    def try_find_223_decomposition(target_rank):
        if time.time() > DEADLINE - 10:
            return None
        
        # Try multiple restarts with different strategies
        max_restarts = 20
        best_solution = None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                np.random.seed(restart * 101 + target_rank)
                x0 = np.random.uniform(-1, 1, size=n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
            elif restart % 3 == 1:
                np.random.seed(restart * 101 + target_rank + 1)
                x0 = np.random.uniform(-2, 2, size=n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
            else:
                np.random.seed(restart * 101 + target_rank + 2)
                # Sparse initialization
                x0 = np.zeros(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
                x0[np.random.choice(len(x0), size=len(x0)//2, replace=False)] = np.random.choice([-1, 1], size=len(x0)//2)
            
            U = x0[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
            V = x0[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
            W = x0[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
            
            # ALS iterations
            for it in range(50):
                if time.time() > DEADLINE - 10:
                    break
                U = als_step_U(V, W, target_rank)
                V = als_step_V(U, W, target_rank)
                W = als_step_W(U, V, target_rank)
                
                err = error_223(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), target_rank)
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-8:
                    break
            
            # Polish with Nelder-Mead if close
            if best_err < 1e-5 and best_solution:
                U0, V0, W0 = best_solution
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                res = minimize(error_223, x0, args=(target_rank,), method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
                if res.fun < 1e-8:
                    U = res.x[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
                    V = res.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
                    W = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
                    
                    U_r = snap_to_half_integers(U)
                    V_r = snap_to_half_integers(V)
                    W_r = snap_to_half_integers(W)
                    
                    if U_r is not None and V_r is not None and W_r is not None:
                        if check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                            return (U_r, V_r, W_r)
        
        return None

    # Try to find rank 9 for 2x2x3 (would give total rank 16)
    sub_decomp = None
    if time.time() < DEADLINE - 25:
        sub_decomp = try_find_223_decomposition(9)
    
    # Try to find rank 10 for 2x2x3 (would give total rank 17)
    if sub_decomp is None and time.time() < DEADLINE - 25:
        sub_decomp = try_find_223_decomposition(10)
        
    if sub_decomp:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy Strassen block (cols 0,1 of B)
        def add_strassen_block(t_offset, k_start):
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
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        
        # Copy sub-decomposition (cols 2,3,4 of B)
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(rank_sub):
                    U[i*m_sub + j][7 + c] = U_sub[i*m_sub + j, c]
                    
        for j in range(m_sub):
            for k_sub in range(p_sub):
                k_global = 2 + k_sub
                for c in range(rank_sub):
                    V[j*p + k_global][7 + c] = V_sub[j*p_sub + k_sub, c]
                    
        for k_sub in range(p_sub):
            k_global = 2 + k_sub
            for i in range(n_sub):
                for c in range(rank_sub):
                    W[k_global*n + i][7 + c] = W_sub[k_sub*n_sub + i, c]
                    
        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_18_solution()

# EVOLVE-BLOCK-END
