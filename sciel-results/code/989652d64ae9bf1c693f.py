"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

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

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

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

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)
    
    # Try alternative: use 1x3 blocks for row 0 with 3x2 blocks
    # This might give a different structure that reduces better
    # But we'll stick with the current structure and rely on better reduction

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    if not verify(U, V, W, col):
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

    def build_M(U_mat, V_mat, W_mat, r):
        M = np.zeros((n * m * p, r))
        for l in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, l] = U_mat[i * m + j, l] * V_mat[j * p + k, l] * W_mat[k * n + i, l]
        return M

    def try_reduce(U_cur, V_cur, W_cur, r):
        if time.time() > _deadline - 15:
            return U_cur, V_cur, W_cur, r
        
        U_next = U_cur
        V_next = V_cur
        W_next = W_cur
        r_next = r
        
        # Iteratively try to find and exploit nullspace vectors
        max_iterations = 20
        for iteration in range(max_iterations):
            if time.time() > _deadline - 15:
                break
            
            M = build_M(U_next, V_next, W_next, r_next)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                null_dim = int(np.sum(s < tol))
                
                if null_dim == 0:
                    break
                
                nullspace = Vt[r_next - null_dim:, :]
                
                # Try each nullspace vector, pick the best elimination
                best_reduction = None
                best_new_rank = r_next
                
                for nv_idx in range(null_dim):
                    null_vec = nullspace[nv_idx, :]
                    if time.time() > _deadline - 15:
                        break
                    
                    # Try eliminating each non-zero component as the pivot
                    abs_nv = np.abs(null_vec)
                    # Sort indices by magnitude descending, try top candidates
                    candidate_indices = np.argsort(-abs_nv)[:min(5, null_dim)]
                    
                    for elim_idx in candidate_indices:
                        if abs(null_vec[elim_idx]) < 1e-10:
                            continue
                        
                        alpha_c = null_vec[elim_idx]
                        new_keep = [c for c in range(r_next) if c != elim_idx]
                        
                        U_test = np.zeros((n * m, len(new_keep)))
                        V_test = np.zeros((m * p, len(new_keep)))
                        W_test = np.zeros((p * n, len(new_keep)))
                        
                        for new_i, old_i in enumerate(new_keep):
                            gamma = 1 - null_vec[old_i] / alpha_c
                            U_test[:, new_i] = U_next[:, old_i] * gamma
                            V_test[:, new_i] = V_next[:, old_i]
                            W_test[:, new_i] = W_next[:, old_i]
                        
                        if verify(U_test, V_test, W_test, len(new_keep)):
                            if len(new_keep) < best_new_rank:
                                best_new_rank = len(new_keep)
                                best_reduction = (U_test, V_test, W_test)
                        
                        # Also try with V scaling
                        U_test2 = np.zeros((n * m, len(new_keep)))
                        V_test2 = np.zeros((m * p, len(new_keep)))
                        W_test2 = np.zeros((p * n, len(new_keep)))
                        
                        for new_i, old_i in enumerate(new_keep):
                            gamma = 1 - null_vec[old_i] / alpha_c
                            U_test2[:, new_i] = U_next[:, old_i]
                            V_test2[:, new_i] = V_next[:, old_i] * gamma
                            W_test2[:, new_i] = W_next[:, old_i]
                        
                        if verify(U_test2, V_test2, W_test2, len(new_keep)):
                            if len(new_keep) < best_new_rank:
                                best_new_rank = len(new_keep)
                                best_reduction = (U_test2, V_test2, W_test2)
                        
                        # Also try with W scaling
                        U_test3 = np.zeros((n * m, len(new_keep)))
                        V_test3 = np.zeros((m * p, len(new_keep)))
                        W_test3 = np.zeros((p * n, len(new_keep)))
                        
                        for new_i, old_i in enumerate(new_keep):
                            gamma = 1 - null_vec[old_i] / alpha_c
                            U_test3[:, new_i] = U_next[:, old_i]
                            V_test3[:, new_i] = V_next[:, old_i]
                            W_test3[:, new_i] = W_next[:, old_i] * gamma
                        
                        if verify(U_test3, V_test3, W_test3, len(new_keep)):
                            if len(new_keep) < best_new_rank:
                                best_new_rank = len(new_keep)
                                best_reduction = (U_test3, V_test3, W_test3)
                
                if best_reduction is None:
                    break
                
                U_next, V_next, W_next = best_reduction
                r_next = best_new_rank
                
            except:
                pass
        
        return U_next, V_next, W_next, r_next

    U_final, V_final, W_final, r_final = try_reduce(U, V, W, col)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
