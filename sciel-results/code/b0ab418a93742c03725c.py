import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7

    def find_223_decomposition(max_iter=5000):
        """Try to find a rank-8 integer/half-integer decomposition for 2x2x3."""
        # Tensor M[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # Need: sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
        # U shape: (4, 8), V shape: (6, 8), W shape: (6, 8)
        
        rank = 8
        U_dim, V_dim, W_dim = 4, 6, 6
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(U_dim, rank)
        V = np.random.randn(V_dim, rank)
        W = np.random.randn(W_dim, rank)
        
        # Build target tensor
        T = np.ones((2, 2, 3))
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 30:
                break
            
            # Update U: for each (i,j), U[i*2+j, :] = T[i,j,:] contracted with V and W
            for i in range(2):
                for j in range(2):
                    # V[j*3+k, r] * W[k*2+i, r] summed over k
                    VW = np.zeros((rank, rank))
                    for k in range(3):
                        VW += np.outer(V[j*3+k, :], W[k*2+i, :])
                    U[i*2+j, :] = np.linalg.solve(VW + 1e-10*np.eye(rank), T[i,j,:].reshape(-1) @ np.ones((3,1)).reshape(-1) * 0 + np.ones(3))
                    # Actually, we need: sum_k sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    # So: sum_r U[i,j,r] * (sum_k V[j,k,r] W[k,i,r]) = 1
                    # Let A_r = sum_k V[j,k,r] W[k,i,r]
                    # Then: sum_r U[i,j,r] * A_r = 1
                    # This is a single equation per (i,j), not a system.
                    
            # Better approach: use ALS with the constraint that the tensor product equals T
            # For fixed V, W: U[i*2+j, r] should satisfy sum_r U[i*2+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1 for all k
            # This is: for each (i,j), U[i*2+j,:] is a vector u such that
            #   sum_k (V[j*3+k,:] * W[k*2+i,:]) * u = 1 (element-wise over k)
            # i.e., sum_r u[r] * V[j*3+k,r] * W[k*2+i,r] = 1 for k=0,1,2
            # This is a 3x8 system: M_{k,r} = V[j*3+k,r] * W[k*2+i,r], solve M @ u = ones(3)
            
            for i in range(2):
                for j in range(2):
                    M_mat = np.zeros((3, rank))
                    for k in range(3):
                        M_mat[k, :] = V[j*3+k, :] * W[k*2+i, :]
                    # Solve M_mat @ u = ones(3) in least squares
                    try:
                        u, _, _, _ = np.linalg.lstsq(M_mat, np.ones(3), rcond=None)
                        U[i*2+j, :] = u
                    except:
                        pass
            
            for j in range(2):
                for k in range(3):
                    M_mat = np.zeros((4, rank))
                    for i in range(2):
                        M_mat[i*2+0, :] = U[i*2+0, :] * W[k*2+i, :]
                        M_mat[i*2+1, :] = U[i*2+1, :] * W[k*2+i, :]
                    # Wait, V[j*3+k, :] should satisfy: sum_i sum_r U[i*2+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1
                    # For fixed j,k: sum_r V[j*3+k,r] * (sum_i U[i*2+j,r] * W[k*2+i,r]) = 1
                    # Let B_r = sum_i U[i*2+j,r] * W[k*2+i,r]
                    # Then sum_r V[j*3+k,r] * B_r = 1
                    B = np.zeros(rank)
                    for i in range(2):
                        B += U[i*2+j, :] * W[k*2+i, :]
                    # V[j*3+k,:] @ B = 1, single equation, underdetermined
                    # Use least squares with minimum norm
                    if np.linalg.norm(B) > 1e-10:
                        V[j*3+k, :] = B / (B @ B)
                    else:
                        V[j*3+k, :] = np.random.randn(rank) * 0.01
            
            for i in range(2):
                for k in range(3):
                    # W[k*2+i, :] should satisfy: sum_j sum_r U[i*2+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1
                    # Let C_r = sum_j U[i*2+j,r] * V[j*3+k,r]
                    # Then sum_r W[k*2+i,r] * C_r = 1
                    C = np.zeros(rank)
                    for j in range(2):
                        C += U[i*2+j, :] * V[j*3+k, :]
                    if np.linalg.norm(C) > 1e-10:
                        W[k*2+i, :] = C / (C @ C)
                    else:
                        W[k*2+i, :] = np.random.randn(rank) * 0.01
        
        # Check accuracy
        max_err = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*2+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return U, V, W, True
        
        return None, None, None, False

    def try_round_to_halfint(U, V, W):
        """Try to round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        max_err = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U_r[i*2+j, :] * V_r[j*3+k, :] * W_r[k*2+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            return U_r, V_r, W_r, True
        return None, None, None, False

    # Try to find 2x2x3 decomposition
    U223, V223, W223, success223 = find_223_decomposition()
    
    if success223:
        U223, V223, W223, rounded = try_round_to_halfint(U223, V223, W223)
    
    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def add_223_block(i_off, j_off, k_off):
        """Add a 2x2x3 block using the precomputed decomposition if available, else schoolbook."""
        if success223 and rounded:
            for r in range(8):
                u = np.zeros(n * m, dtype=float)
                v = np.zeros(m * p, dtype=float)
                w = np.zeros(p * n, dtype=float)
                for ii in range(2):
                    for jj in range(2):
                        u[(i_off+ii)*m + (j_off+jj)] = U223[ii*2+jj, r]
                for jj in range(2):
                    for kk in range(3):
                        v[(j_off+jj)*p + (k_off+kk)] = V223[jj*3+kk, r]
                for kk in range(3):
                    for ii in range(2):
                        w[(k_off+kk)*n + (i_off+ii)] = W223[kk*2+ii, r]
                U_vecs.append(u)
                V_vecs.append(v)
                W_vecs.append(w)
        else:
            # Schoolbook fallback: 2*2*3 = 12 terms
            for ii in range(2):
                for jj in range(2):
                    for kk in range(3):
                        add_rank1_term([(i_off+ii)*m+(j_off+jj)], [1.0],
                                      [(j_off+jj)*p+(k_off+kk)], [1.0],
                                      [(k_off+kk)*n+(i_off+ii)], [1.0])

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+i], [1.0])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1.0], [j*p+kk], [1.0], [kk*n+ii], [1.0])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+k], [1.0], [k*n+ii], [1.0])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off + 1
            j0 = j_off; j1 = j_off + 1
            k0 = k_off; k1 = k_off + 1

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

        if n_l == 2 and m_l == 2 and p_l == 3:
            add_223_block(i_off, j_off, k_off)
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
            # Split as 2x2x3 + 2x2x2 + 2x2x2
            gen_terms(2, 2, 3, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 3)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 5)
            return

        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    gen_terms(n, m, p, 0, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Verify
    rank = len(U_vecs)
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = np.sum(U[ci*m + cj] * V[cj*p + ck] * W[ck*n + ci])
                if abs(prod_sum - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to pure schoolbook
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
