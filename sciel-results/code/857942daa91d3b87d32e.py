"""Matrix multiplication 4,4,9 tensor decomposition - Strassen-based with targeted search."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 145
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
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target, atol=1e-10)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
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
        """Build a 4x4x4 Strassen decomposition using 2x2 blocks."""
        terms = []
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                terms.append((u_vec, v_vec, w_vec))
        
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
        all_terms.extend(build_strassen_4x4(0, 0, 0, 0, 0, 0))
        all_terms.extend(build_strassen_4x4(0, 0, 0, 4, 0, 4))
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
    rank = len(all_terms)
    U = np.zeros((n_dim, rank))
    V = np.zeros((m_dim, rank))
    W = np.zeros((p_dim, rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
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
    
    # Try to find a better solution using structured search
    def try_structured_search(target_rank, max_iters=100):
        """Try to find a decomposition using structured supports from Strassen patterns."""
        if target_rank >= rank:
            return None
        
        import random
        rng = random.Random(42 + target_rank)
        
        # Pre-compute supports from Strassen solution to bias the search
        strassen_supports_U = [np.nonzero(U[:, r])[0] for r in range(rank)]
        strassen_supports_V = [np.nonzero(V[:, r])[0] for r in range(rank)]
        strassen_supports_W = [np.nonzero(W[:, r])[0] for r in range(rank)]
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            U_new = np.zeros((n_dim, target_rank))
            V_new = np.zeros((m_dim, target_rank))
            W_new = np.zeros((p_dim, target_rank))
            
            # Generate structured supports: bias towards Strassen supports
            for r in range(target_rank):
                # 50% chance to use a Strassen-like support, 50% random
                if rng.random() < 0.5:
                    # Pick a random Strassen term to mimic
                    ref_r = rng.randint(0, rank-1)
                    u_support = list(strassen_supports_U[ref_r])
                    v_support = list(strassen_supports_V[ref_r])
                    w_support = list(strassen_supports_W[ref_r])
                    
                    # Perturb: swap 1-2 indices
                    if u_support and rng.random() < 0.5:
                        idx = rng.randint(0, len(u_support)-1)
                        u_support[idx] = rng.randint(0, n_dim-1)
                    if v_support and rng.random() < 0.5:
                        idx = rng.randint(0, len(v_support)-1)
                        v_support[idx] = rng.randint(0, m_dim-1)
                    if w_support and rng.random() < 0.5:
                        idx = rng.randint(0, len(w_support)-1)
                        w_support[idx] = rng.randint(0, p_dim-1)
                else:
                    u_support = rng.sample(range(n_dim), rng.randint(2, 4))
                    v_support = rng.sample(range(m_dim), rng.randint(2, 4))
                    w_support = rng.sample(range(p_dim), rng.randint(2, 4))
                
                for pos in u_support:
                    U_new[pos, r] = rng.choice([-1, 1])
                for pos in v_support:
                    V_new[pos, r] = rng.choice([-1, 1])
                for pos in w_support:
                    W_new[pos, r] = rng.choice([-1, 1])
            
            # Solve for W given U and V
            D = np.zeros((n_dim * m_dim, target_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U_new[a, :] * V_new[b, :]
            
            T_reshaped = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except Exception:
                continue
            
            # Check if solution is exact
            T_check = np.einsum('ar,br,cr->abc', U_new, V_new, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                # Check for rational entries
                all_vals = np.concatenate([U_new.flatten(), V_new.flatten(), W_sol.flatten()])
                for denom in [1, 2, 4, 8, 16, 32]:
                    scaled = all_vals * denom
                    if np.allclose(scaled, np.round(scaled), atol=1e-6):
                        U_exact = np.round(U_new * denom) / float(denom)
                        V_exact = np.round(V_new * denom) / float(denom)
                        W_exact = np.round(W_sol * denom) / float(denom)
                        T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                        if np.allclose(T_final.flatten(), target, atol=1e-10):
                            return U_exact, V_exact, W_exact, target_rank
        
        return None
    
    # Try a block-structured search exploiting the 4x4x9 rectangular structure
    def try_block_structured_search(target_rank, max_iters=50):
        """
        Try to find a decomposition by modifying the schoolbook tail (last 16 terms)
        with Strassen-like patterns that exploit the rectangular structure.
        
        The idea: instead of 16 schoolbook terms for the 4x4x1 part, try to use
        fewer terms with more complex supports that cover multiple (i,j,k) positions.
        """
        if target_rank >= rank:
            return None
        
        import random
        rng = random.Random(100 + target_rank)
        
        # The first 98 terms are the Strassen 4x4x4 parts (fixed)
        # The last 16 terms are the schoolbook 4x4x1 part
        # We'll try to replace some of the last 16 with more efficient patterns
        
        strassen_U = U[:, :98].copy()
        strassen_V = V[:, :98].copy()
        strassen_W = W[:, :98].copy()
        
        # The schoolbook part covers (i,j,k=8) for all i,j in 0..3
        # Each term has U[i*4+j]=1, V[j*9+8]=1, W[8*4+i]=1
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Try different numbers of tail terms
            num_tail = target_rank - 98
            if num_tail < 8 or num_tail > 16:
                continue
            
            U_new = np.zeros((n_dim, target_rank))
            V_new = np.zeros((m_dim, target_rank))
            W_new = np.zeros((p_dim, target_rank))
            
            # Copy Strassen part
            U_new[:, :98] = strassen_U
            V_new[:, :98] = strassen_V
            W_new[:, :98] = strassen_W
            
            # Generate tail terms
            for r in range(num_tail):
                col = 98 + r
                
                # Each tail term should cover some subset of the 16 (i,j) positions
                # for the k=8 column
                # Strategy: each term covers 2-4 (i,j) pairs with appropriate W entries
                
                # Choose 2-4 (i,j) pairs
                num_pairs = rng.randint(2, min(4, 16 // num_tail + 1))
                pairs = rng.sample([(i,j) for i in range(4) for j in range(4)], num_pairs)
                
                for (i, j) in pairs:
                    u_idx = i * m + j
                    v_idx = j * p + 8
                    w_idx = 8 * n + i
                    
                    # Assign coefficients (try small integers)
                    coeff = rng.choice([-2, -1, 1, 2])
                    U_new[u_idx, col] += coeff
                    V_new[v_idx, col] += coeff
                    W_new[w_idx, col] += coeff
            
            # Now solve for the tail W values to correct any errors
            # First, compute what the Strassen part gives
            T_strassen = np.einsum('ar,br,cr->abc', strassen_U, strassen_V, strassen_W)
            
            # The target has 1s at (i,j,8) for all i,j
            # Compute the residual
            residual = np.zeros((n_dim, m_dim, p_dim))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        residual[a_idx, b_idx, c_idx] = target[a_idx * m_dim * p_dim + b_idx * p_dim + c_idx] - T_strassen[a_idx, b_idx, c_idx]
            
            # The tail terms should cancel the residual
            # For each tail term r, we have U_new[:, 98+r], V_new[:, 98+r], W_new[:, 98+r]
            # We need: sum_r U_new[a,98+r] * V_new[b,98+r] * W_new[c,98+r] = residual[a,b,c]
            
            # This is a linear system in W_new[:, 98:]
            # For each (a,b,c): sum_r (U_new[a,98+r] * V_new[b,98+r]) * W_new[c,98+r] = residual[a,b,c]
            
            # Build the system
            A_sys = np.zeros((n_dim * m_dim * p_dim, num_tail))
            b_vec = np.zeros(n_dim * m_dim * p_dim)
            
            for a in range(n_dim):
                for b in range(m_dim):
                    for c in range(p_dim):
                        flat = a * m_dim * p_dim + b * p_dim + c
                        for r in range(num_tail):
                            A_sys[flat, r] = U_new[a, 98+r] * V_new[b, 98+r]
                        b_vec[flat] = residual[a, b, c]
            
            # Solve
            try:
                W_tail, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_vec, rcond=None)
            except Exception:
                continue
            
            W_new[:, 98:] = W_tail
            
            # Check if exact
            T_check = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                # Check for rational entries
                all_vals = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
                for denom in [1, 2, 4, 8, 16, 32, 64]:
                    scaled = all_vals * denom
                    if np.allclose(scaled, np.round(scaled), atol=1e-6):
                        U_exact = np.round(U_new * denom) / float(denom)
                        V_exact = np.round(V_new * denom) / float(denom)
                        W_exact = np.round(W_new * denom) / float(denom)
                        T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                        if np.allclose(T_final.flatten(), target, atol=1e-10):
                            return U_exact, V_exact, W_exact, target_rank
        
        return None
    
    # Try ranks below 114
    best_result = None
    for target_rank in [113, 112, 111]:
        if time.time() > deadline - 20:
            break
        
        # Try block-structured search first (more targeted)
        result = try_block_structured_search(target_rank, max_iters=50)
        if result is not None:
            U, V, W, r = result
            best_result = (U, V, W, r)
            break
        
        # Then try general structured search
        result = try_structured_search(target_rank, max_iters=50)
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
