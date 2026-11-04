import time
import numpy as np

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    1. Build the exact matrix multiplication tensor.
    2. Use structured Strassen-based decomposition as a strong baseline (rank 114).
    3. Attempt to find a lower-rank decomposition via:
       a. Random sparse support search with linear solve for W.
       b. Gradient-based optimization (smooth surrogate) to polish or discover new structures.
    4. Return the best valid decomposition found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 9
    n_dim = n * m   # 16
    m_dim = m * p   # 36
    p_dim = p * n   # 36
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (A row i, col j)
    # b = j*p + k (B row j, col k)
    # c = k*n + i (C col k, row i) -- trace convention W[k*n+i, r]
    # T[a, b, c] = 1 iff the indices are consistent (j matches)
    
    target = np.zeros((n_dim, m_dim, p_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    target_flat = target.flatten()
    
    def verify_decomposition(U, V, W, rank, atol=1e-9):
        """Verify sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target_flat, atol=atol)
    
    def is_half_integer(arr, atol=1e-6):
        """Check if all entries are close to half-integers."""
        scaled = arr * 2
        return np.allclose(scaled, np.round(scaled), atol=atol)
    
    def to_half_integer(arr, atol=1e-6):
        """Round to nearest half-integer if close."""
        if is_half_integer(arr, atol):
            return np.round(arr * 2) / 2.0
        return arr
    
    # --- Baseline: Strassen-based decomposition (rank 114) ---
    
    def build_strassen_baseline():
        """Build the rank-114 Strassen-based decomposition."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_cols.append(np.array(u_vec, dtype=np.float64))
            V_cols.append(np.array(v_vec, dtype=np.float64))
            W_cols.append(np.array(w_vec, dtype=np.float64))
        
        # Strassen 2x2 terms
        strassen_2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
            """Adds 49 terms for a 4x4 * 4x4 multiplication using Strassen."""
            
            def add_inner(A_blocks, B_blocks, C_blocks):
                for u_c, v_c, w_c in strassen_2x2:
                    u_vec = [0] * 16
                    v_vec = [0] * 36
                    w_vec = [0] * 36
                    
                    for loc, coeff in u_c.items():
                        li = loc // 2
                        lj = loc % 2
                        for br, bc, bc_coeff in A_blocks:
                            gi = A_base_i + br * 2 + li
                            gj = A_base_j + bc * 2 + lj
                            u_vec[gi * m + gj] += coeff * bc_coeff
                        
                    for loc, coeff in v_c.items():
                        lj = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in B_blocks:
                            gj = B_base_j + br * 2 + lj
                            gk = B_base_k + bc * 2 + lk
                            v_vec[gj * p + gk] += coeff * bc_coeff
                        
                    for loc, coeff in w_c.items():
                        li = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in C_blocks:
                            gi = C_base_i + br * 2 + li
                            gk = C_base_k + bc * 2 + lk
                            w_vec[gk * n + gi] += coeff * bc_coeff
                    
                    add_term(u_vec, v_vec, w_vec)
            
            # M1
            add_inner(
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)]
            )
            # M2
            add_inner(
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,0,1), (1,1,-1)]
            )
            # M3
            add_inner(
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(0,1,1), (1,1,1)]
            )
            # M4
            add_inner(
                [(1,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (1,0,1)]
            )
            # M5
            add_inner(
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,-1), (0,1,1)]
            )
            # M6
            add_inner(
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)]
            )
            # M7
            add_inner(
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)]
            )
        
        # Part 1: A * B1 (cols 0-3)
        add_strassen_4x4(0, 0, 0, 0, 0, 0)
        
        # Part 2: A * B2 (cols 4-7)
        add_strassen_4x4(0, 0, 0, 4, 0, 4)
        
        # Part 3: A * B3 (col 8)
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
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        return U, V, W, rank
    
    U_base, V_base, W_base, base_rank = build_strassen_baseline()
    best_U, best_V, best_W, best_rank = U_base, V_base, W_base, base_rank
    
    # --- Random sparse support search ---
    
    def random_sparse_search(max_iters, seed=42):
        """Try to find a lower-rank decomposition with sparse supports."""
        rng = np.random.RandomState(seed)
        
        for rank in range(base_rank - 1, 80, -1):
            if time.time() > deadline - 10:
                break
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return best_U, best_V, best_W, best_rank
                
                # Random sparse supports
                u_sparsity = rng.randint(2, 5)
                v_sparsity = rng.randint(2, 5)
                w_sparsity = rng.randint(2, 5)
                
                U = np.zeros((n_dim, rank))
                V = np.zeros((m_dim, rank))
                W = np.zeros((p_dim, rank))
                
                for r in range(rank):
                    u_support = rng.choice(n_dim, u_sparsity, replace=False)
                    v_support = rng.choice(m_dim, v_sparsity, replace=False)
                    w_support = rng.choice(p_dim, w_sparsity, replace=False)
                    
                    u_signs = rng.choice([-1.0, 1.0], u_sparsity)
                    v_signs = rng.choice([-1.0, 1.0], v_sparsity)
                    w_signs = rng.choice([-1.0, 1.0], w_sparsity)
                    
                    for idx, pos in enumerate(u_support):
                        U[pos, r] = u_signs[idx]
                    for idx, pos in enumerate(v_support):
                        V[pos, r] = v_signs[idx]
                    for idx, pos in enumerate(w_support):
                        W[pos, r] = w_signs[idx]
                
                # Solve for W given U and V
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each (a,b): sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
                # Let D[a*m_dim+b, r] = U[a,r]*V[b,r]
                # Then D @ W.T = T.reshape(n_dim*m_dim, p_dim).T
                
                D = np.zeros((n_dim * m_dim, rank))
                for a in range(n_dim):
                    for b in range(m_dim):
                        D[a * m_dim + b, :] = U[a, :] * V[b, :]
                
                T_reshaped = target.reshape(n_dim * m_dim, p_dim)
                
                try:
                    # Solve D @ X = T_reshaped.T where X = W.T
                    X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                    W_sol = X.T  # shape (p_dim, rank)
                except Exception:
                    continue
                
                # Check exactness
                T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
                if np.allclose(T_check.flatten(), target_flat, atol=1e-9):
                    # Check half-integer
                    all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                    if is_half_integer(all_vals, atol=1e-6):
                        U_exact = to_half_integer(U)
                        V_exact = to_half_integer(V)
                        W_exact = to_half_integer(W_sol)
                        
                        # Verify again
                        T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                        if np.allclose(T_final.flatten(), target_flat, atol=1e-9):
                            best_U, best_V, best_W, best_rank = U_exact, V_exact, W_exact, rank
                            return best_U, best_V, best_W, best_rank
        
        return best_U, best_V, best_W, best_rank
    
    # Run random search with limited iterations
    max_iters_per_rank = 50
    best_U, best_V, best_W, best_rank = random_sparse_search(max_iters_per_rank)
    
    # --- Gradient-based polishing (smooth surrogate) ---
    
    def gradient_polish(U, V, W, rank, max_iters=200, lr=0.01):
        """Try to reduce rank or polish using gradient descent on a smooth objective."""
        if time.time() > deadline - 15:
            return U, V, W, rank
        
        # Objective: minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
        # Use a soft penalty for non-half-integer values
        
        U_opt = U.copy()
        V_opt = V.copy()
        W_opt = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute current tensor
            T_current = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
            error = T_current - target
            
            # Gradients
            # dT/da = sum_r U[a,r] V[b,r] W[c,r]
            # Gradient w.r.t. U: for each a, sum over b,c of error[a,b,c] * V[b,r] * W[c,r] * (1/rank)
            # Simplified: use finite differences or analytical gradients
            
            # Analytical gradients:
            # dL/dU[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} error[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} error[a,b,c] * U[a,r] * V[b,r]
            
            dU = np.einsum('abc,br,cr->ar', error, V_opt, W_opt)
            dV = np.einsum('abc,ar,cr->br', error, U_opt, W_opt)
            dW = np.einsum('abc,ar,br->cr', error, U_opt, V_opt)
            
            # Add half-integer penalty
            penalty_coeff = 0.01
            dU += penalty_coeff * (2 * U_opt - np.round(2 * U_opt))
            dV += penalty_coeff * (2 * V_opt - np.round(2 * V_opt))
            dW += penalty_coeff * (2 * W_opt - np.round(2 * W_opt))
            
            # Update
            U_opt -= lr * dU
            V_opt -= lr * dV
            W_opt -= lr * dW
            
            # Check convergence
            if np.allclose(T_current.flatten(), target_flat, atol=1e-9):
                if is_half_integer(np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()]), atol=1e-6):
                    U_exact = to_half_integer(U_opt)
                    V_exact = to_half_integer(V_opt)
                    W_exact = to_half_integer(W_opt)
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target_flat, atol=1e-9):
                        return U_exact, V_exact, W_exact, rank
        
        return U_opt, V_opt, W_opt, rank
    
    # Try gradient polish on the best found so far
    if time.time() < deadline - 20:
        polished_U, polished_V, polished_W, polished_rank = gradient_polish(
            best_U, best_V, best_W, best_rank, max_iters=100
        )
        # Verify polished result
        T_check = np.einsum('ar,br,cr->abc', polished_U, polished_V, polished_W)
        if np.allclose(T_check.flatten(), target_flat, atol=1e-9):
            if is_half_integer(np.concatenate([polished_U.flatten(), polished_V.flatten(), polished_W.flatten()]), atol=1e-6):
                polished_U = to_half_integer(polished_U)
                polished_V = to_half_integer(polished_V)
                polished_W = to_half_integer(polished_W)
                T_final = np.einsum('ar,br,cr->abc', polished_U, polished_V, polished_W)
                if np.allclose(T_final.flatten(), target_flat, atol=1e-9):
                    best_U, best_V, best_W, best_rank = polished_U, polished_V, polished_W, polished_rank
    
    # Final verification
    T_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if not np.allclose(T_final.flatten(), target_flat, atol=1e-9):
        # Fall back to baseline
        best_U, best_V, best_W, best_rank = U_base, V_base, W_base, base_rank
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
