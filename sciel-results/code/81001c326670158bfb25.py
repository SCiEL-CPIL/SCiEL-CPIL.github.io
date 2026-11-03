import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7
    # Current best known structured rank is 78.

    U_vecs = []
    V_vecs = []
    W_vecs = []

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

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off
            j = j_off
            k = k_off
            add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1], [jj*p+kk], [1], [kk*n+i], [1])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1], [j*p+kk], [1], [kk*n+ii], [1])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1], [jj*p+k], [1], [k*n+ii], [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            k1 = k_off + 1

            # Strassen 2x2x2
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])
            return

        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return

        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return

        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return

        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    gen_terms(n, m, p, 0, 0, 0)

    # Convert to float for output
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T

    # Peeling: try to reduce rank by finding linearly dependent rank-1 components
    # We look for subsets of rank-1 terms that can be combined
    # Strategy: check if any rank-1 term is a linear combination of others
    # For each term t, check if U[:,t] is in span of other U columns
    # This is expensive, so we do it with a time budget
    
    rank = U.shape[1]
    if time.time() < DEADLINE - 20 and rank > 20:
        # Try to peel: find if any column is linearly dependent
        # Use a greedy approach: try removing terms one at a time
        # and check if the decomposition still works
        
        def verify_decomposition(U_mat, V_mat, W_mat):
            """Check if the decomposition is valid"""
            U_3d = U_mat.reshape(n, m, -1)
            V_3d = V_mat.reshape(m, p, -1)
            W_3d = W_mat.reshape(p, n, -1)
            T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
            return np.allclose(T, 1.0, atol=1e-10)
        
        # Try removing each term and see if it still works
        # This is O(rank^2 * verification_cost) which may be too slow
        # So we limit the number of attempts
        
        max_attempts = 50
        attempts = 0
        removed = set()
        
        for t in range(rank):
            if time.time() > DEADLINE - 15:
                break
            if t in removed:
                continue
            if attempts >= max_attempts:
                break
            
            attempts += 1
            # Try removing term t
            mask = [i for i in range(rank) if i != t and i not in removed]
            if len(mask) < 2:
                continue
            
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if verify_decomposition(U_test, V_test, W_test):
                removed.add(t)
        
        if removed:
            final_mask = [i for i in range(rank) if i not in removed]
            U = U[:, final_mask]
            V = V[:, final_mask]
            W = W[:, final_mask]
            rank = U.shape[1]

    # Final verification
    try:
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback to schoolbook
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array([[float(x) for x in row] for row in U_vecs]).T
            V = np.array([[float(x) for x in row] for row in V_vecs]).T
            W = np.array([[float(x) for x in row] for row in W_vecs]).T
    except Exception:
        # Fallback to schoolbook on error
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array([[float(x) for x in row] for row in U_vecs]).T
        V = np.array([[float(x) for x in row] for row in V_vecs]).T
        W = np.array([[float(x) for x in row] for row in W_vecs]).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
