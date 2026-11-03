import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    n, m, p = 3, 4, 7
    # Build a decomposition using tensor power of Strassen (k=2 gives 4x4x4, 49 terms)
    # Then project to 3x4x7 by zeroing out the 4th row/col where needed.
    # For 3x4x7: we need i in 0..2, j in 0..3, k in 0..6
    # The 4x4x4 tensor power covers i,j,k in 0..3
    # We can use the 4x4x4 decomposition (49 terms) and just keep the terms
    # that contribute to our 3x4x7 subproblem.
    # Actually, the 4x4x4 multiplication tensor has 4*4*4=64 entries.
    # Our 3x4x7 tensor has 3*4*7=84 entries.
    # The 4x4x4 Strassen^2 decomposition gives rank 49 for 4x4x4.
    # For 3x4x7, we can embed it in 4x4x8 (rank would be 49*2=98?) or 4x4x7.
    # Better: use the 4x4x4 decomposition and note that 3x4x7 is a sub-tensor of 4x4x8.
    # 4x4x8 = 4x4x4 + 4x4x4, so rank <= 49+49 = 98.
    # But 3x4x7 is a sub-tensor of 4x4x8, so rank(3x4x7) <= 98.
    # The additive block approach gives 78, which is better.
    
    # Let me try a different approach: use the known rank-23 decomposition for 3x3x3.
    # 3x4x7 = 3x3x7 + 3x1x7
    # 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1 = 23 + 23 + 9 = 55
    # 3x1x7 = 21
    # Total = 76
    
    # I need a rank-23 decomposition of 3x3x3. Let me use a known one.
    # The following is a rank-23 decomposition from the literature.
    # Each term is (u, v, w) where u, v, w are 9-element vectors (for 3x3).
    # I'll use the decomposition from B. Le Gall or a similar source.
    
    # Actually, let me just use the additive block decomposition with optimization.
    # The current approach gives 78. Let me see if I can do better with a smarter split.
    
    # Alternative: 3x4x7 = (2x4x7) + (1x4x7)
    # 2x4x7: use Strassen on 2x2 blocks
    # 2x4x7 = (2x2x7) + (2x2x7)
    # 2x2x7 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x1) = 7+7+7+4 = 25
    # So 2x4x7 = 50
    # 1x4x7 = 28
    # Total = 78
    
    # What if I use a 3x3x3 block?
    # 3x4x7 = (3x3x7) + (3x1x7)
    # If I have rank 23 for 3x3x3, then:
    # 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1 = 23+23+9 = 55
    # 3x1x7 = 21
    # Total = 76
    
    # Let me hardcode a rank-23 decomposition for 3x3x3.
    # I'll use the one from https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    # or construct it from known results.
    
    # For now, let me stick with the additive block approach but try to optimize
    # the splitting strategy more carefully.
    
    def peel_terms(U_vecs, V_vecs, W_vecs, n, m, p, deadline):
        """Iteratively peel off linearly dependent rank-1 terms."""
        rank = len(U_vecs)
        if rank <= 1:
            return U_vecs, V_vecs, W_vecs
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            rank = len(U_vecs)
            
            U_mat = np.array([[float(x) for x in row] for row in U_vecs]).T
            V_mat = np.array([[float(x) for x in row] for row in V_vecs]).T
            W_mat = np.array([[float(x) for x in row] for row in W_vecs]).T
            
            try:
                U_rank = np.linalg.matrix_rank(U_mat)
                if U_rank < rank:
                    U_T = U_mat.T
                    U_svd = np.linalg.svd(U_T, full_matrices=True)
                    U_Vt = U_svd[2]
                    null_start = U_rank
                    null_vecs = U_Vt[null_start:]
                    
                    for nv in null_vecs:
                        if time.time() > deadline:
                            return U_vecs, V_vecs, W_vecs
                        
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < 1e-10:
                            continue
                        
                        V_check = np.sum(nv[:, None] * V_mat, axis=1)
                        W_check = np.sum(nv[:, None] * W_mat, axis=1)
                        
                        if np.max(np.abs(V_check)) > 1e-8 or np.max(np.abs(W_check)) > 1e-8:
                            continue
                        
                        scale_factors = []
                        for t in range(rank):
                            if t == max_idx:
                                continue
                            scale = 1 + nv[t] / nv[max_idx]
                            if abs(scale) < 1e-10:
                                continue
                            scale_factors.append((t, scale))
                        
                        if len(scale_factors) < rank - 1:
                            final_U = []
                            final_V = []
                            final_W = []
                            for t, scale in scale_factors:
                                final_U.append([x * scale for x in U_vecs[t]])
                                final_V.append([x * scale for x in V_vecs[t]])
                                final_W.append([x * scale for x in W_vecs[t]])
                            
                            valid = True
                            for ci in range(n):
                                for cj in range(m):
                                    for ck in range(p):
                                        prod_sum = sum(
                                            final_U[t][ci*m + cj] * final_V[t][cj*p + ck] * final_W[t][ck*n + ci]
                                            for t in range(len(final_U))
                                        )
                                        if abs(prod_sum - 1) > 1e-6:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                U_vecs = final_U
                                V_vecs = final_V
                                W_vecs = final_W
                                changed = True
                                break
                        else:
                            continue
            except:
                pass
        
        return U_vecs, V_vecs, W_vecs

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)
        return (u, v, w)
    
    def gen_block(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate rank-1 terms for an n_l x m_l x p_l block."""
        terms = []
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            terms.append(add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1]))
            return terms
        
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off + 1
            j0 = j_off; j1 = j_off + 1
            k0 = k_off; k1 = k_off + 1
            for (ui, uv, vi, vv, wi, wv) in strassen_terms:
                u_idx = [i*m+j for i, j in zip([0,1,2,3], [0,0,1,1])]
                # Map Strassen indices to actual positions
                u_map = {0: i0*m+j0, 1: i0*m+j1, 2: i1*m+j0, 3: i1*m+j1}
                v_map = {0: j0*p+k0, 1: j0*p+k1, 2: j1*p+k0, 3: j1*p+k1, 4: j1*p+k0, 5: j1*p+k1}
                # Actually let me just use the explicit form
                pass
            
            # Use explicit Strassen terms
            s = [
                ([i0*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0], [1],
                 [k0*n+i1, k1*n+i1], [1, -1]),
                ([i0*m+j0], [1],
                 [j0*p+k1, j1*p+k1], [1, -1],
                 [k1*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j1], [1],
                 [j1*p+k0, j0*p+k0], [1, -1],
                 [k0*n+i0, k0*n+i1], [1, 1]),
                ([i0*m+j0, i0*m+j1], [1, 1],
                 [j1*p+k1], [1],
                 [k0*n+i0, k1*n+i0], [-1, 1]),
                ([i1*m+j0, i0*m+j0], [1, -1],
                 [j0*p+k0, j0*p+k1], [1, 1],
                 [k1*n+i1], [1]),
                ([i0*m+j1, i1*m+j1], [1, -1],
                 [j1*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0], [1]),
            ]
            for (ui, uv, vi, vv, wi, wv) in s:
                terms.append(add_rank1_term(ui, uv, vi, vv, wi, wv))
            return terms
        
        # Recursive split: try splitting along the largest dimension
        # Split n if n_l >= 2
        if n_l >= 2:
            n1 = n_l // 2
            n2 = n_l - n1
            terms.extend(gen_block(n1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n2, m_l, p_l, i_off + n1, j_off, k_off))
            return terms
        
        if m_l >= 2:
            m1 = m_l // 2
            m2 = m_l - m1
            terms.extend(gen_block(n_l, m1, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m2, p_l, i_off, j_off + m1, k_off))
            return terms
        
        if p_l >= 2:
            p1 = p_l // 2
            p2 = p_l - p1
            terms.extend(gen_block(n_l, m_l, p1, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m_l, p2, i_off, j_off, k_off + p1))
            return terms
        
        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i; jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
        return terms
    
    def gen_block(n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate rank-1 terms for an n_l x m_l x p_l block."""
        terms = []
        
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            terms.append(add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1]))
            return terms
        
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([i_off*m+jj], [1], [jj*p+kk], [1], [kk*n+i_off], [1]))
            return terms
        
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+j_off], [1], [j_off*p+kk], [1], [kk*n+ii], [1]))
            return terms
        
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+k_off], [1], [k_off*n+ii], [1]))
            return terms
        
        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off + 1
            j0 = j_off; j1 = j_off + 1
            k0 = k_off; k1 = k_off + 1
            s = [
                ([i0*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j0, i1*m+j1], [1, 1],
                 [j0*p+k0], [1],
                 [k0*n+i1, k1*n+i1], [1, -1]),
                ([i0*m+j0], [1],
                 [j0*p+k1, j1*p+k1], [1, -1],
                 [k1*n+i0, k1*n+i1], [1, 1]),
                ([i1*m+j1], [1],
                 [j1*p+k0, j0*p+k0], [1, -1],
                 [k0*n+i0, k0*n+i1], [1, 1]),
                ([i0*m+j0, i0*m+j1], [1, 1],
                 [j1*p+k1], [1],
                 [k0*n+i0, k1*n+i0], [-1, 1]),
                ([i1*m+j0, i0*m+j0], [1, -1],
                 [j0*p+k0, j0*p+k1], [1, 1],
                 [k1*n+i1], [1]),
                ([i0*m+j1, i1*m+j1], [1, -1],
                 [j1*p+k0, j1*p+k1], [1, 1],
                 [k0*n+i0], [1]),
            ]
            for (ui, uv, vi, vv, wi, wv) in s:
                terms.append(add_rank1_term(ui, uv, vi, vv, wi, wv))
            return terms
        
        # Recursive split: try splitting along the largest dimension
        if n_l >= 2:
            n1 = n_l // 2
            n2 = n_l - n1
            terms.extend(gen_block(n1, m_l, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n2, m_l, p_l, i_off + n1, j_off, k_off))
            return terms
        
        if m_l >= 2:
            m1 = m_l // 2
            m2 = m_l - m1
            terms.extend(gen_block(n_l, m1, p_l, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m2, p_l, i_off, j_off + m1, k_off))
            return terms
        
        if p_l >= 2:
            p1 = p_l // 2
            p2 = p_l - p1
            terms.extend(gen_block(n_l, m_l, p1, i_off, j_off, k_off))
            terms.extend(gen_block(n_l, m_l, p2, i_off, j_off, k_off + p1))
            return terms
        
        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i; jj = j_off + j; kk = k_off + k
                    terms.append(add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1]))
        return terms
    
    all_terms = gen_block(n, m, p, 0, 0, 0)
    
    U_vecs = [t[0] for t in all_terms]
    V_vecs = [t[1] for t in all_terms]
    W_vecs = [t[2] for t in all_terms]
    
    # Verify
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if prod_sum != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    term = add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
                    U_vecs.append(term[0])
                    V_vecs.append(term[1])
                    W_vecs.append(term[2])
    
    # Try to reduce rank by peeling
    if time.time() < DEADLINE - 10:
        U_vecs, V_vecs, W_vecs = peel_terms(U_vecs, V_vecs, W_vecs, n, m, p, DEADLINE - 10)
    
    # Final verification
    final_valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if abs(float(prod_sum) - 1.0) > 1e-6:
                    final_valid = False
                    break
            if not final_valid:
                break
        if not final_valid:
            break
    
    if not final_valid:
        # Fallback to the original decomposition
        all_terms_orig = gen_optimal(n, m, p, 0, 0, 0)
        U_vecs = [t[0] for t in all_terms_orig]
        V_vecs = [t[1] for t in all_terms_orig]
        W_vecs = [t[2] for t in all_terms_orig]
    
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
