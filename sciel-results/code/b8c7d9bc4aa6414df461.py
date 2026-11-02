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
            j_idx = i % m_sub
            i_local = i // m_sub
            
            H = np.zeros((rank, rank))
            G = np.zeros(rank)
            
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j_idx == jp else 0.0
                    v_vec = V[jp*p_sub + k, :]
                    w_vec = W[k*n_sub + i_local, :]
                    
                    v_w = v_vec * w_vec
                    H += np.outer(v_w, v_w)
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
        
        max_restarts = 30
        best_solution = None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            np.random.seed(restart * 101 + target_rank)
            
            # Sparse random initialization with half-integers
            size = n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank
            x0 = np.zeros(size)
            # Set 50% of entries to random half-integers
            num_non_zero = size // 2
            indices = np.random.choice(size, size=num_non_zero, replace=False)
            x0[indices] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=num_non_zero)
            
            U = x0[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
            V = x0[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
            W = x0[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
            
            # ALS iterations
            for it in range(100):
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
            if best_err < 1e-4 and best_solution:
                U0, V0, W0 = best_solution
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                res = minimize(error_223, x0, args=(target_rank,), method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
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
