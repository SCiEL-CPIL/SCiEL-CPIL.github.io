"""Matrix multiplication 4,4,9 tensor decomposition - Strassen-based with targeted search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the target tensor for verification
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target, atol=1e-10)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
    # Each term is (u_dict, v_dict, w_dict) where keys are (row, col) indices
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def build_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Build a 4x4x4 Strassen decomposition using 2x2 blocks. Returns list of (u_vec, v_vec, w_vec)."""
        terms = []
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Fill u_vec from A_blocks
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # Fill v_vec from B_blocks
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # Fill w_vec from C_blocks
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                terms.append((u_vec, v_vec, w_vec))
        
        # 7 Strassen combinations for 4x4 using 2x2 blocks
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        return terms
    
    def construct_strassen_114():
        """Standard Strassen-based construction: rank 49 + 49 + 16 = 114."""
        all_terms = []
        
        # First 4x4 block of B: columns 0-3
        all_terms.extend(build_strassen_4x4(0, 0, 0, 0, 0, 0))
        
        # Second 4x4 block of B: columns 4-7
        all_terms.extend(build_strassen_4x4(0, 0, 0, 4, 0, 4))
        
        # Last column of B: column 8, use schoolbook
        for i in range(n):
            for j in range(m):
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                all_terms.append((u_vec, v_vec, w_vec))
        
        return all_terms
    
    # Build the standard rank-114 solution
    all_terms = construct_strassen_114()
    
    # Verify the standard solution
    rank = len(all_terms)
    U = np.zeros((n_dim, rank))
    V = np.zeros((m_dim, rank))
    W = np.zeros((p_dim, rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        for i in range(n_dim):
            if u_vec[i] != 0:
                U[i, r] = u_vec[i]
        for i in range(m_dim):
            if v_vec[i] != 0:
                V[i, r] = v_vec[i]
        for i in range(p_dim):
            if w_vec[i] != 0:
                W[i, r] = w_vec[i]
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n_dim, rank_sb))
        V_sb = np.zeros((m_dim, rank_sb))
        W_sb = np.zeros((p_dim, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now try to find a better solution using targeted search
    # Strategy: Try to merge some of the rank-1 terms or find a lower-rank decomposition
    
    # Approach: Use a more sophisticated search
    # Fix the structure of U and V (sparse, based on Strassen patterns) and solve for W
    
    def try_targeted_search(target_rank, max_iters=20):
        """Try to find a decomposition of the given rank."""
        if target_rank >= rank:
            return None
        
        import random
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Create sparse random supports based on Strassen-like patterns
            U = np.zeros((n_dim, target_rank))
            V = np.zeros((m_dim, target_rank))
            W = np.zeros((p_dim, target_rank))
            
            for r in range(target_rank):
                # Each column of U has ~2-3 nonzeros
                u_count = rng.randint(2, 3)
                u_support = rng.sample(range(n_dim), u_count)
                for pos in u_support:
                    U[pos, r] = rng.choice([-1, 1])
                
                v_count = rng.randint(2, 3)
                v_support = rng.sample(range(m_dim), v_count)
                for pos in v_support:
                    V[pos, r] = rng.choice([-1, 1])
                
                w_count = rng.randint(2, 3)
                w_support = rng.sample(range(p_dim), w_count)
                for pos in w_support:
                    W[pos, r] = rng.choice([-1, 1])
            
            # Solve for W given U and V
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed U,V: W @ (U[:,a]*V[:,b]) = T[a,b,:]
            D = np.zeros((n_dim * m_dim, target_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U[a, :] * V[b, :]
            
            T_reshaped = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except Exception:
                continue
            
            # Check if solution is exact and has half-integer entries
            T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-6):
                    U_exact = np.round(U * 2) / 2.0
                    V_exact = np.round(V * 2) / 2.0
                    W_exact = np.round(W_sol * 2) / 2.0
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target, atol=1e-10):
                        return U_exact, V_exact, W_exact, target_rank
        
        return None
    
    # Try a few ranks below 114
    best_result = None
    for target_rank in [113, 112, 111, 110]:
        if time.time() > deadline - 20:
            break
        result = try_targeted_search(target_rank, max_iters=10)
        if result is not None:
            U, V, W, r = result
            best_result = (U, V, W, r)
            break
    
    if best_result is not None:
        U, V, W, rank_out = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Return the rank-114 Strassen solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
