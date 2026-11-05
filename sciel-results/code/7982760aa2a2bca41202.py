import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    def build_decomposition(i_blocks, j_blocks, k_blocks):
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_vec, v_vec, w_vec):
            u_col = np.zeros(n_m)
            v_col = np.zeros(m_p)
            w_col = np.zeros(p_n)
            for idx, val in u_vec:
                u_col[idx] = val
            for idx, val in v_vec:
                v_col[idx] = val
            for idx, val in w_vec:
                w_col[idx] = val
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
        
        def schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off):
            for i_loc in range(n_prime):
                for j_loc in range(m_prime):
                    for k_loc in range(p_prime):
                        ig = i_off + i_loc
                        jg = j_off + j_loc
                        kg = k_off + k_loc
                        I = ig * m + jg
                        J = jg * p + kg
                        K = kg * n + ig
                        add_term([(I, 1)], [(J, 1)], [(K, 1)])
        
        def strassen_2x2x2(i_off, j_off, k_off):
            def gI(i_l, j_l): return (i_off + i_l) * m + (j_off + j_l)
            def gJ(j_l, k_l): return (j_off + j_l) * p + (k_off + k_l)
            def gK(k_l, i_l): return (k_off + k_l) * n + (i_off + i_l)
            
            add_term(
                [(gI(0,0), 1), (gI(1,1), 1)],
                [(gJ(0,0), 1), (gJ(1,1), 1)],
                [(gK(0,0), 1), (gK(1,1), 1)]
            )
            add_term(
                [(gI(1,0), 1), (gI(1,1), 1)],
                [(gJ(0,0), 1)],
                [(gK(0,1), 1), (gK(1,1), -1)]
            )
            add_term(
                [(gI(0,0), 1)],
                [(gJ(0,1), 1), (gJ(1,1), -1)],
                [(gK(1,0), 1), (gK(1,1), 1)]
            )
            add_term(
                [(gI(1,1), 1)],
                [(gJ(1,0), 1), (gJ(0,0), -1)],
                [(gK(0,0), 1), (gK(0,1), 1)]
            )
            add_term(
                [(gI(0,0), 1), (gI(0,1), 1)],
                [(gJ(1,1), 1)],
                [(gK(0,0), -1), (gK(1,0), 1)]
            )
            add_term(
                [(gI(1,0), 1), (gI(0,0), -1)],
                [(gJ(0,0), 1), (gJ(0,1), 1)],
                [(gK(1,1), 1)]
            )
            add_term(
                [(gI(0,1), 1), (gI(1,1), -1)],
                [(gJ(1,0), 1), (gJ(1,1), 1)],
                [(gK(0,0), 1)]
            )
        
        def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
            if i_sz == 2 and j_sz == 2 and k_sz == 2:
                strassen_2x2x2(i_off, j_off, k_off)
            elif i_sz == 2 and j_sz == 2 and k_sz == 3:
                strassen_2x2x2(i_off, j_off, k_off)
                schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
            elif i_sz == 2 and j_sz == 3 and k_sz == 2:
                strassen_2x2x2(i_off, j_off, k_off)
                schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
            elif i_sz == 2 and j_sz == 3 and k_sz == 3:
                strassen_2x2x2(i_off, j_off, k_off)
                schoolbook(2, 2, 1, i_off, j_off, k_off + 2)
                schoolbook(2, 1, 2, i_off, j_off + 2, k_off)
                schoolbook(2, 1, 1, i_off, j_off + 2, k_off + 2)
            else:
                schoolbook(i_sz, j_sz, k_sz, i_off, j_off, k_off)
        
        for i_off, i_sz in i_blocks:
            for j_off, j_sz in j_blocks:
                for k_off, k_sz in k_blocks:
                    handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
        
        if not U_cols:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        # Verify
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return None
        
        return U, V, W, rank
    
    def try_reduce_rank(U, V, W, rank):
        """Try to reduce rank by finding a subset of terms that sums to target."""
        if rank <= 1:
            return U, V, W, rank
        
        # Build target tensor
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        # Each term is a rank-1 tensor
        term_vecs = []
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            t_c = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            term_vecs.append(t_c.flatten())
        
        M = np.array(term_vecs)  # shape (rank, n*m*p)
        
        # We want to find a subset S such that sum_{c in S} M[c] = target_flat
        # This is a subset sum problem in high dimensions, which is NP-hard.
        # Instead, try: find the null space and see if we can zero out some terms.
        # Or: use integer programming heuristic.
        
        # Simple heuristic: try removing one term at a time and check if the rest can compensate
        # via scaling. But scaling must preserve integer/half-integer entries.
        
        # Alternative: check if any term is zero or duplicate
        for c in range(rank):
            if np.allclose(M[c], 0, atol=1e-9):
                # Remove this term
                U_new = np.delete(U, c, axis=1)
                V_new = np.delete(V, c, axis=1)
                W_new = np.delete(W, c, axis=1)
                return U_new, V_new, W_new, rank - 1
        
        # Try to find linear dependencies: if term_c = sum_{c' != c} alpha_{c'} term_{c'}
        # with alpha in {0, 1, -1}, we can replace.
        # This is expensive, so only try for small rank.
        
        return U, V, W, rank
    
    # Try the main partition: n=[2,2], m=[2,3], p=[2,3]
    partitions = [
        ([(0, 2), (2, 2)], [(0, 2), (2, 3)], [(0, 2), (2, 3)]),
        ([(0, 2), (2, 2)], [(0, 3), (3, 2)], [(0, 2), (2, 3)]),
        ([(0, 2), (2, 2)], [(0, 2), (2, 3)], [(0, 3), (3, 2)]),
        ([(0, 3), (3, 1)], [(0, 2), (2, 3)], [(0, 2), (2, 3)]),
        ([(0, 1), (1, 3)], [(0, 2), (2, 3)], [(0, 2), (2, 3)]),
    ]
    
    for i_blocks, j_blocks, k_blocks in partitions:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = build_decomposition(i_blocks, j_blocks, k_blocks)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W, rank)
        
        # Try to reduce
        if best_result is not None:
            U, V, W, rank = best_result
            if time.time() - start_time < DEADLINE - 20:
                U2, V2, W2, rank2 = try_reduce_rank(U, V, W, rank)
                if rank2 < best_rank:
                    best_rank = rank2
                    best_result = (U2, V2, W2, rank2)
    
    if best_result is None:
        # Schoolbook fallback
        rank = n * m * p
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        W = np.zeros((p_n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
