"""Matrix multiplication 4,4,9 tensor decomposition - Strassen-based with targeted search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Build target tensor
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
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Each term is (U_dict, V_dict, W_dict) with keys (row, col) and values
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def build_2x2x9_terms(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Build 2x2 x 2x9 terms using Strassen for first 2 cols, schoolbook for rest."""
        terms = []
        
        # For columns 0,1 of B (relative): use Strassen 2x2x2
        for u_c, v_c, w_c in strassen_2x2:
            u_vec = {}
            v_vec = {}
            w_vec = {}
            
            for (loc_i, loc_j), coeff in u_c.items():
                gi = A_i_base + loc_i
                gj = A_j_base + loc_j
                u_vec[(gi, gj)] = u_vec.get((gi, gj), 0) + coeff
            
            for (loc_j, loc_k), coeff in v_c.items():
                gj = B_j_base + loc_j
                gk = B_k_base + loc_k  # k = 0,1
                v_vec[(gj, gk)] = v_vec.get((gj, gk), 0) + coeff
            
            for (loc_i, loc_k), coeff in w_c.items():
                gi = C_i_base + loc_i
                gk = C_k_base + loc_k  # k = 0,1
                w_vec[(gk, gi)] = w_vec.get((gk, gi), 0) + coeff
            
            terms.append((u_vec, v_vec, w_vec))
        
        # For columns 2-8 of B (relative): schoolbook, 7 terms
        for k_rel in range(2, 9):
            for i_rel in range(2):
                for j_rel in range(2):
                    u_vec = {((A_i_base + i_rel), (A_j_base + j_rel)): 1}
                    v_vec = {((B_j_base + j_rel), (B_k_base + k_rel)): 1}
                    w_vec = {((C_k_base + k_rel), (C_i_base + i_rel)): 1}
                    terms.append((u_vec, v_vec, w_vec))
        
        return terms
    
    def build_4x4x9():
        """Build 4x4 x 4x9 using 2x2 blocks."""
        all_terms = []
        
        # A is 4x4, split into 2x2 blocks: (i_block, j_block) where i_block, j_block in {0,1}
        # B is 4x9, split into 2x2 blocks for rows: (j_block, k_block) where j_block in {0,1}, k_block in {0,1,2,3,4}
        # But 9 = 2+2+2+2+1, so we have 4 full 2-col blocks and 1 single col
        
        # Actually, let's be more systematic:
        # A[i,j] where i=0..3, j=0..3
        # B[j,k] where j=0..3, k=0..8
        # C[i,k] where i=0..3, k=0..8
        
        # For each pair of 2x2 A-blocks and 2x2 B-blocks (sharing j dimension):
        # A block (i0, j0) times B block (j0, k0) contributes to C block (i0, k0)
        
        # i0 in {0,1}, j0 in {0,1}, k0 in {0,1,2,3,4} (for 2-col blocks) plus k=8 single
        
        for i0 in range(2):
            for j0 in range(2):
                for k0 in range(4):  # k0 = 0,1,2,3 -> columns 2*k0, 2*k0+1
                    terms = build_2x2x9_terms(
                        i0 * 2, j0 * 2,  # A block base
                        j0 * 2, k0 * 2,  # B block base
                        i0 * 2, k0 * 2   # C block base
                    )
                    all_terms.extend(terms)
        
        # Handle the 9th column (k=8) separately: schoolbook for all i,j
        for i in range(4):
            for j in range(4):
                u_vec = {((i), (j)): 1}
                v_vec = {((j), (8)): 1}
                w_vec = {((8), (i)): 1}
                all_terms.append((u_vec, v_vec, w_vec))
        
        return all_terms
    
    # Build the decomposition
    all_terms = build_4x4x9()
    rank = len(all_terms)
    
    U = np.zeros((n_dim, rank))
    V = np.zeros((m_dim, rank))
    W = np.zeros((p_dim, rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        for (gi, gj), coeff in u_vec.items():
            U[gi * m + gj, r] += coeff
        for (gj, gk), coeff in v_vec.items():
            V[gj * p + gk, r] += coeff
        for (gk, gi), coeff in w_vec.items():
            W[gk * n + gi, r] += coeff
    
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
    
    # Try to reduce rank by removing terms
    def try_remove_term(U_cur, V_cur, W_cur, cur_rank, term_idx):
        if time.time() > deadline - 20:
            return None
        
        keep = [i for i in range(cur_rank) if i != term_idx]
        U_sub = U_cur[:, keep]
        V_sub = V_cur[:, keep]
        
        D = np.zeros((n_dim * m_dim, cur_rank - 1))
        for a in range(n_dim):
            for b in range(m_dim):
                D[a * m_dim + b, :] = U_sub[a, :] * V_sub[b, :]
        
        T_reshaped = target.reshape(n_dim * m_dim, p_dim)
        
        try:
            X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
            W_sol = X.T
        except Exception:
            return None
        
        T_check = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sol)
        if not np.allclose(T_check.flatten(), target, atol=1e-8):
            return None
        
        all_vals = np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sol.flatten()])
        for denom in [1, 2, 4, 8, 16, 32]:
            scaled = all_vals * denom
            if np.allclose(scaled, np.round(scaled), atol=1e-5):
                U_exact = np.round(U_sub * denom) / float(denom)
                V_exact = np.round(V_sub * denom) / float(denom)
                W_exact = np.round(W_sol * denom) / float(denom)
                T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                if np.allclose(T_final.flatten(), target, atol=1e-10):
                    return U_exact, V_exact, W_exact, cur_rank - 1
        return None
    
    best_result = None
    best_rank = rank
    
    # Try removing terms
    for term_idx in range(rank):
        if time.time() > deadline - 30:
            break
        result = try_remove_term(U, V, W, rank, term_idx)
        if result is not None:
            best_result = result
            best_rank = result[3]
            break
    
    if best_result is not None:
        U, V, W, rank_out = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
