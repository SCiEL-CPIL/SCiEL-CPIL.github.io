"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
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

    # Sub-problem: 2x2x3
    n_sub, m_sub, p_sub = 2, 2, 3
    
    def check_exact_223(U, V, W, rank, tol=1e-5):
        # U: (4, rank), V: (6, rank), W: (6, rank)
        # Target: T[i, j, j', k, i', k'] = delta(i, i') delta(k, k') delta(j, j')
        # C[i', k'] = sum_c W[i', k', c] (sum_{i, j} U[i, j, c] A[i, j]) (sum_{j', k} V[j', k, c] B[j', k])
        # We check the coefficients.
        for i in range(n_sub):
            for k in range(p_sub):
                # Compute the matrix of coefficients for A[i, j] B[j', k]
                # Actually, we just check the tensor reconstruction.
                for j in range(m_sub):
                    for jp in range(m_sub):
                        target = 1.0 if (j == jp) else 0.0
                        # Wait, the target is delta(j, jp) for the term A[i, j] B[jp, k] to contribute to C[i, k]?
                        # C[i, k] = sum_j A[i, j] B[j, k].
                        # So coefficient of A[i, j] B[jp, k] is 1 if j == jp, else 0.
                        val = 0.0
                        for c in range(rank):
                            # U index: i * m_sub + j
                            # V index: jp * p_sub + k
                            # W index: k * n_sub + i
                            val += U[i * m_sub + j, c] * V[jp * p_sub + k, c] * W[k * n_sub + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def als_223(rank, max_iters=100, tol=1e-8):
        # U: (4, rank), V: (6, rank), W: (6, rank)
        # Target tensor T_flat size 144
        # T[i, j, jp, k, i, k] -> delta(i, i) is 1, delta(k, k) is 1, delta(j, jp) is 1.
        
        # Construct Target Vector
        T_target = np.zeros(144)
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        if j == jp:
                            # Index in flat array:
                            # i (2), j (2), jp (2), k (3), i (2), k (3)
                            # Order: i, j, jp, k, i, k
                            idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                            T_target[idx] = 1.0

        def get_A_matrix(U_or_V_or_W, other1, other2, dim_self, dim_other1, dim_other2):
            # Constructs matrix A such that A @ vec(self) = vec(self @ other1 @ other2)
            # self shape: (dim_self, rank) -> we solve for one column at a time?
            # No, we solve for the whole matrix at once?
            # ALS usually solves for one factor at a time.
            # Let's solve for U (4 x rank).
            # We want to minimize || T - sum_c U_c V_c W_c ||^2
            # This is equivalent to solving a linear system for each column of U?
            # No, the columns of U are coupled? No, they are independent in the sum.
            # But V and W are fixed.
            # So for each c, we have a linear system for U_c?
            # Yes.
            pass

        # Simpler ALS:
        # For each c, solve for U_c, V_c, W_c sequentially?
        # No, standard ALS:
        # 1. Fix V, W. Solve for U (all columns).
        #    The objective is sum_c || T - sum_{c'} U_{c'} V_{c'} W_{c'} ||^2
        #    This is not separable in c if we consider the sum.
        #    But if we fix all other c' != c, we can solve for U_c.
        #    So we iterate c=0..rank-1, solving for U_c.
        
        def solve_for_U(V, W):
            U = np.zeros((n_sub * m_sub, rank))
            for c in range(rank):
                # Residual T_res = T_target - sum_{c' != c} U_{c'} V_{c'} W_{c'}
                T_res = T_target.copy()
                for c_prime in range(rank):
                    if c_prime == c: continue
                    # Subtract U_{c_prime} V_{c_prime} W_{c_prime}
                    for i in range(n_sub):
                        for j in range(m_sub):
                            u_val = U[i * m_sub + j, c_prime]
                            if u_val == 0: continue
                            for jp in range(m_sub):
                                for k in range(p_sub):
                                    v_val = V[jp * p_sub + k, c_prime]
                                    if v_val == 0: continue
                                    w_val = W[k * n_sub + i, c_prime]
                                    if w_val == 0: continue
                                    val = u_val * v_val * w_val
                                    if val != 0:
                                        idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                        T_res[idx] -= val
                                        
                # Solve A @ u_c = T_res
                # A is 144 x 4
                A = np.zeros((144, n_sub * m_sub))
                for i in range(n_sub):
                    for j in range(m_sub):
                        u_idx = i * m_sub + j
                        for jp in range(m_sub):
                            for k in range(p_sub):
                                v_val = V[jp * p_sub + k, c]
                                w_val = W[k * n_sub + i, c]
                                val = v_val * w_val
                                if val != 0:
                                    idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                    A[idx, u_idx] = val
                                    
                # Least squares
                u_c, _, _, _ = np.linalg.lstsq(A, T_res, rcond=None)
                U[:, c] = u_c
            return U

        # Similar for V and W
        def solve_for_V(U, W):
            V = np.zeros((m_sub * p_sub, rank))
            for c in range(rank):
                T_res = T_target.copy()
                for c_prime in range(rank):
                    if c_prime == c: continue
                    for i in range(n_sub):
                        for j in range(m_sub):
                            u_val = U[i * m_sub + j, c_prime]
                            if u_val == 0: continue
                            for jp in range(m_sub):
                                for k in range(p_sub):
                                    v_val = V[jp * p_sub + k, c_prime]
                                    if v_val == 0: continue
                                    w_val = W[k * n_sub + i, c_prime]
                                    if w_val == 0: continue
                                    val = u_val * v_val * w_val
                                    if val != 0:
                                        idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                        T_res[idx] -= val
                                        
                A = np.zeros((144, m_sub * p_sub))
                for jp in range(m_sub):
                    for k in range(p_sub):
                        v_idx = jp * p_sub + k
                        for i in range(n_sub):
                            for j in range(m_sub):
                                u_val = U[i * m_sub + j, c]
                                w_val = W[k * n_sub + i, c]
                                val = u_val * w_val
                                if val != 0:
                                    idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                    A[idx, v_idx] = val
                                    
                v_c, _, _, _ = np.linalg.lstsq(A, T_res, rcond=None)
                V[:, c] = v_c
            return V

        def solve_for_W(U, V):
            W = np.zeros((p_sub * n_sub, rank))
            for c in range(rank):
                T_res = T_target.copy()
                for c_prime in range(rank):
                    if c_prime == c: continue
                    for i in range(n_sub):
                        for j in range(m_sub):
                            u_val = U[i * m_sub + j, c_prime]
                            if u_val == 0: continue
                            for jp in range(m_sub):
                                for k in range(p_sub):
                                    v_val = V[jp * p_sub + k, c_prime]
                                    if v_val == 0: continue
                                    w_val = W[k * n_sub + i, c_prime]
                                    if w_val == 0: continue
                                    val = u_val * v_val * w_val
                                    if val != 0:
                                        idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                        T_res[idx] -= val
                                        
                A = np.zeros((144, p_sub * n_sub))
                for k in range(p_sub):
                    for i in range(n_sub):
                        w_idx = k * n_sub + i
                        for j in range(m_sub):
                            for jp in range(m_sub):
                                u_val = U[i * m_sub + j, c]
                                v_val = V[jp * p_sub + k, c]
                                val = u_val * v_val
                                if val != 0:
                                    idx = (((((i * m_sub) + j) * m_sub + jp) * p_sub + k) * n_sub + i) * p_sub + k
                                    A[idx, w_idx] = val
                                    
                w_c, _, _, _ = np.linalg.lstsq(A, T_res, rcond=None)
                W[:, c] = w_c
            return W

        # Main ALS loop
        for restart in range(20):
            if time.time() > DEADLINE - 10:
                break
            np.random.seed(restart * 123 + rank)
            U = np.random.normal(size=(n_sub * m_sub, rank))
            V = np.random.normal(size=(m_sub * p_sub, rank))
            W = np.random.normal(size=(p_sub * n_sub, rank))
            
            for it in range(max_iters):
                if time.time() > DEADLINE - 10:
                    break
                U_new = solve_for_U(V, W)
                V_new = solve_for_V(U_new, W)
                W_new = solve_for_W(U_new, V_new)
                
                # Check convergence
                diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
                U, V, W = U_new, V_new, W_new
                if diff < tol:
                    break
                    
            # Check exactness
            if check_exact_223(U, V, W, rank):
                U_r = snap_to_half_integers(U)
                V_r = snap_to_half_integers(V)
                W_r = snap_to_half_integers(W)
                if U_r is not None and V_r is not None and W_r is not None:
                    if check_exact_223(U_r, V_r, W_r, rank):
                        return (U_r, V_r, W_r)
                
                # Try integer snap
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if check_exact_223(U_i, V_i, W_i, rank):
                    return (U_i, V_i, W_i)
                    
        return None

    # Try to find rank 10 for 2x2x3
    sub_decomp = None
    if time.time() < DEADLINE - 20:
        sub_decomp = als_223(10)
        
    if sub_decomp:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
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
