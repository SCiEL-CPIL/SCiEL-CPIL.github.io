"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_decomposition():
        """Build a block-structured rank-66 decomposition."""
        U66 = [[0.0] * 66 for _ in range(n * m)]
        V66 = [[0.0] * 66 for _ in range(m * p)]
        W66 = [[0.0] * 66 for _ in range(p * n)]
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
            U66[A00][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1; V66[B11][col] += 1
            W66[C00][col] += 1; W66[C11][col] += 1
            col += 1

            # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
            U66[A10][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1
            W66[C10][col] += 1; W66[C11][col] -= 1
            col += 1

            # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
            U66[A00][col] += 1
            V66[B01][col] += 1; V66[B11][col] -= 1
            W66[C01][col] += 1; W66[C11][col] += 1
            col += 1

            # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
            U66[A11][col] += 1
            V66[B10][col] += 1; V66[B00][col] -= 1
            W66[C00][col] += 1; W66[C10][col] += 1
            col += 1

            # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
            U66[A00][col] += 1; U66[A01][col] += 1
            V66[B11][col] += 1
            W66[C00][col] -= 1; W66[C01][col] += 1
            col += 1

            # M6 = (A10-A00)*(B00+B01) -> C11 += M6
            U66[A10][col] += 1; U66[A00][col] -= 1
            V66[B00][col] += 1; V66[B01][col] += 1
            W66[C11][col] += 1
            col += 1

            # M7 = (A01-A11)*(B10+B11) -> C00 += M7
            U66[A01][col] += 1; U66[A11][col] -= 1
            V66[B10][col] += 1; V66[B11][col] += 1
            W66[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            # Direct: A0*B00 -> C0, A1*B10 -> C0
            U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
            col += 1
            # Direct: A0*B01 -> C1, A1*B11 -> C1
            U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
            col += 1

        # Build the decomposition
        # Row 0: direct terms for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)

        # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)
        return U, V, W

    def try_reduce_rank(U, V, W, r, max_reductions=50):
        """Try to reduce rank by finding linear dependencies via SVD null space."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_reductions):
            if time.time() > _deadline - 15:
                break
                
            nmp = n * m * p
            # Build the tensor product matrix M where M[ijk, l] = U[ij,l]*V[jk,l]*W[ki,l]
            M = np.zeros((nmp, best_rank))
            for l in range(best_rank):
                ul = best_U[:, l]  # shape (n*m,)
                vl = best_V[:, l]  # shape (m*p,)
                wl = best_W[:, l]  # shape (p*n,)
                # M[i*m*p + j*p + k, l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                for i in range(n):
                    for j in range(m):
                        u_val = ul[i * m + j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = vl[j * p + k]
                            if v_val == 0:
                                continue
                            w_val = wl[k * n + i]
                            if w_val == 0:
                                continue
                            M[i * m * p + j * p + k, l] = u_val * v_val * w_val
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = int(np.sum(s > tol))
                
                if rank_M < best_rank:
                    # Null space vectors are in the last (best_rank - rank_M) rows of Vt
                    null_space = Vt[rank_M:, :].T  # shape (best_rank - rank_M, best_rank)
                    
                    found = False
                    for nv in null_space:
                        if time.time() > _deadline - 15:
                            break
                        
                        # Try each column index as the one to eliminate
                        # Prefer columns with large |nv| component
                        indices = np.argsort(np.abs(nv))[::-1]
                        for elim_idx in indices:
                            if abs(nv[elim_idx]) < tol:
                                continue
                            
                            U_new = np.delete(best_U, elim_idx, axis=1)
                            V_new = np.delete(best_V, elim_idx, axis=1)
                            W_new = np.delete(best_W, elim_idx, axis=1)
                            
                            if verify(U_new, V_new, W_new, best_rank - 1):
                                best_U, best_V, best_W = U_new, V_new, W_new
                                best_rank -= 1
                                found = True
                                break
                        
                        if found:
                            break
                    
                    if not found:
                        break
                else:
                    break
            except:
                break
        
        return best_rank, best_U, best_V, best_W

    # Build initial decomposition
    U, V, W = build_decomposition()
    
    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    best_rank, best_U, best_V, best_W = try_reduce_rank(U, V, W, 66, max_reductions=50)
    
    if time.time() > _deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
