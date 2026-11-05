"""Matrix multiplication 4,4,9 tensor decomposition - Strassen hybrid with local search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Build target tensor
    target = np.zeros((n_dim, m_dim, p_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    target_flat = target.flatten()
    
    # Strassen-based construction
    def construct_strassen():
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        strassen_2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        def add_strassen_4x4(B_col_offset):
            def add_inner(A_blocks, B_blocks, C_blocks):
                for u_c, v_c, w_c in strassen_2x2:
                    u_vec = [0] * 16
                    v_vec = [0] * 36
                    w_vec = [0] * 36
                    
                    for loc, coeff in u_c.items():
                        li = loc // 2
                        lj = loc % 2
                        for br, bc, bc_coeff in A_blocks:
                            gi = br * 2 + li
                            gj = bc * 2 + lj
                            u_vec[gi * m + gj] += coeff * bc_coeff
                    
                    for loc, coeff in v_c.items():
                        lj = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in B_blocks:
                            gj = br * 2 + lj
                            gk = B_col_offset + bc * 2 + lk
                            v_vec[gj * p + gk] += coeff * bc_coeff
                    
                    for loc, coeff in w_c.items():
                        li = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in C_blocks:
                            gi = br * 2 + li
                            gk = B_col_offset + bc * 2 + lk
                            w_vec[gk * n + gi] += coeff * bc_coeff
                    
                    add_term(u_vec, v_vec, w_vec)
            
            add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
            add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
            add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
            add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
            add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
            add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
            add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        add_strassen_4x4(0)
        add_strassen_4x4(4)
        
        for i in range(4):
            for j in range(4):
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
        
        rank = len(U_cols)
        U = np.array([[U_cols[col][row] for col in range(rank)] for row in range(n * m)], dtype=float)
        V = np.array([[V_cols[col][row] for col in range(rank)] for row in range(m * p)], dtype=float)
        W = np.array([[W_cols[col][row] for col in range(rank)] for row in range(p * n)], dtype=float)
        return U, V, W, rank
    
    U, V, W, rank = construct_strassen()
    
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target_flat, atol=1e-9)
    
    # Local search: try to reduce rank by merging columns
    def try_reduce_rank(U, V, W, rank, max_attempts=200):
        """Try to reduce rank by finding linearly dependent columns."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Try to find two columns that are linearly related
            # If column r1 and r2 are proportional, we can merge them
            r1, r2 = random.sample(range(rank), 2)
            
            # Check if U[:,r1] and U[:,r2] are proportional
            u1, u2 = U[:, r1], U[:, r2]
            v1, v2 = V[:, r1], V[:, r2]
            w1, w2 = W[:, r1], W[:, r2]
            
            # Try to find alpha such that u2 = alpha * u1
            nonzero_u1 = np.nonzero(u1)[0]
            if len(nonzero_u1) == 0:
                continue
            
            # If u1 and u2 have different support, they can't be proportional
            nonzero_u2 = np.nonzero(u2)[0]
            if set(nonzero_u1) != set(nonzero_u2):
                continue
            
            # Compute alpha
            idx = nonzero_u1[0]
            if u1[idx] == 0:
                continue
            alpha = u2[idx] / u1[idx]
            
            # Check if u2 = alpha * u1
            if not np.allclose(u2, alpha * u1, atol=1e-9):
                continue
            
            # Similarly for v
            nonzero_v1 = np.nonzero(v1)[0]
            if len(nonzero_v1) == 0:
                continue
            nonzero_v2 = np.nonzero(v2)[0]
            if set(nonzero_v1) != set(nonzero_v2):
                continue
            idx = nonzero_v1[0]
            if v1[idx] == 0:
                continue
            alpha_v = v2[idx] / v1[idx]
            if not np.allclose(v2, alpha_v * v1, atol=1e-9):
                continue
            if alpha_v != alpha:
                continue
            
            # Similarly for w
            nonzero_w1 = np.nonzero(w1)[0]
            if len(nonzero_w1) == 0:
                continue
            nonzero_w2 = np.nonzero(w2)[0]
            if set(nonzero_w1) != set(nonzero_w2):
                continue
            idx = nonzero_w1[0]
            if w1[idx] == 0:
                continue
            alpha_w = w2[idx] / w1[idx]
            if not np.allclose(w2, alpha_w * w1, atol=1e-9):
                continue
            if alpha_w != alpha:
                continue
            
            # All three are proportional with same alpha - we can merge
            # Remove column r2
            mask = np.ones(rank, dtype=bool)
            mask[r2] = False
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if verify(U_new, V_new, W_new):
                return U_new, V_new, W_new, rank - 1
        
        return U, V, W, rank
    
    # Try to reduce rank
    best_U, best_V, best_W, best_rank = U, V, W, rank
    for _ in range(3):
        if time.time() > deadline - 15:
            break
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try random perturbation + re-solve
    def try_perturb_solve(U, V, W, rank, n_iters=30):
        """Perturb supports and re-solve for W."""
        rng = random.Random(42)
        for it in range(n_iters):
            if time.time() > deadline - 10:
                break
            
            U_trial = U.copy()
            V_trial = V.copy()
            
            # Perturb a few entries
            for _ in range(random.randint(1, 3)):
                mat = random.choice([U_trial, V_trial])
                row = random.randint(0, mat.shape[0] - 1)
                col = random.randint(0, rank - 1)
                old_val = mat[row, col]
                # Try setting to 0 or 1 or -1
                mat[row, col] = random.choice([0, 1, -1])
                if mat[row, col] == old_val:
                    mat[row, col] = 0 if old_val != 0 else 1
            
            # Build D matrix
            D = np.zeros((n_dim * m_dim, rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U_trial[a, :] * V_trial[b, :]
            
            T_reshaped = target_flat.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_trial = X.T
            except Exception:
                continue
            
            # Check if exact
            T_check = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
            if np.allclose(T_check.flatten(), target_flat, atol=1e-8):
                # Check for half-integer values
                all_vals = np.concatenate([U_trial.flatten(), V_trial.flatten(), W_trial.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-5):
                    U_exact = np.round(U_trial * 2) / 2.0
                    V_exact = np.round(V_trial * 2) / 2.0
                    W_exact = np.round(W_trial * 2) / 2.0
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target_flat, atol=1e-10):
                        return U_exact, V_exact, W_exact, rank
        
        return None
    
    # Try perturbation
    if time.time() < deadline - 20:
        result = try_perturb_solve(U, V, W, rank)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try reducing further
    if time.time() < deadline - 10:
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    U_out = [[best_U[row, col] for col in range(best_rank)] for row in range(n_dim)]
    V_out = [[best_V[row, col] for col in range(best_rank)] for row in range(m_dim)]
    W_out = [[best_W[row, col] for col in range(best_rank)] for row in range(p_dim)]
    return {"U": U_out, "V": V_out, "W": W_out}
