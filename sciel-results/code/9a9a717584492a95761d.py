import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7
    # If we had a better decomposition, we would use it here.
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

    # Fast verification using numpy
    # Construct the tensor from U, V, W
    # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # We check if T is the identity tensor for matrix multiplication.
    # Identity tensor: T[i,j,k] = 1 if (i,j,k) corresponds to a product term?
    # No, the tensor is $M_{3,4,7}$.
    # $M_{i,j,k} = 1$ if $i \in [0,3), j \in [0,4), k \in [0,7)$.
    # Actually, the tensor is defined by $M_{i,j,k} = \delta_{i,i} \delta_{j,j} \delta_{k,k}$?
    # No, the tensor for matrix multiplication $C = AB$ is $T_{i,j,k} = 1$ for all valid $i,j,k$.
    # Wait, the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # So $T_{i,j,k} = 1$ for all $i,j,k$.
    
    # Let's verify using numpy einsum
    # U shape (n*m, rank) -> (n, m, rank)
    # V shape (m*p, rank) -> (m, p, rank)
    # W shape (p*n, rank) -> (p, n, rank)
    
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # Compute sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # This is a contraction.
    # Result should be 1 for all i,j,k.
    
    # Use np.einsum
    # 'ijt,jkt,kit->ijk'
    # But indices are:
    # U: i, j, t
    # V: j, k, t
    # W: k, i, t
    # Sum over t.
    
    try:
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        # Check if T is all ones
        if not np.allclose(T, 1.0):
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
