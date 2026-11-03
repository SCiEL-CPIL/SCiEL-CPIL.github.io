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

    if not verify(U, V, W, 66):
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

    def try_reduce(U_cur, V_cur, W_cur, r):
        if time.time() > _deadline - 10:
            return U_cur, V_cur, W_cur, r
        
        # Build flattening matrix
        M = np.zeros((n * m * p, r))
        for l in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, l] = U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
        
        # Find nullspace
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            null_dim = np.sum(s < tol)
            
            if null_dim == 0:
                return U_cur, V_cur, W_cur, r
            
            # Get nullspace vectors
            nullspace = Vt[r - null_dim:, :]
            
            # Try to eliminate columns using nullspace vectors
            # We iterate through nullspace vectors and try to eliminate one column per vector
            # We need to be careful: eliminating one column changes the matrix M for the next step.
            # So we do it sequentially.
            
            U_next = U_cur
            V_next = V_cur
            W_next = W_cur
            r_next = r
            
            # We'll keep a list of columns to keep
            keep_cols = list(range(r))
            
            for null_vec in nullspace:
                if time.time() > _deadline - 10:
                    break
                
                # Find the index with the largest component
                max_idx = np.argmax(np.abs(null_vec))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue
                
                # Check if this column is still in our keep list
                if max_idx not in keep_cols:
                    continue
                
                # Calculate scaling factors for the other columns
                alpha_c = null_vec[max_idx]
                
                # New columns are keep_cols except max_idx
                new_keep = [c for c in keep_cols if c != max_idx]
                
                U_test = np.zeros((n * m, len(new_keep)))
                V_test = np.zeros((m * p, len(new_keep)))
                W_test = np.zeros((p * n, len(new_keep)))
                
                for new_i, old_i in enumerate(new_keep):
                    gamma = 1 - null_vec[old_i] / alpha_c
                    U_test[:, new_i] = U_cur[:, old_i] * gamma
                    V_test[:, new_i] = V_cur[:, old_i]
                    W_test[:, new_i] = W_cur[:, old_i]
                
                # Verify
                if verify(U_test, V_test, W_test, len(new_keep)):
                    U_next = U_test
                    V_next = V_test
                    W_next = W_test
                    r_next = len(new_keep)
                    keep_cols = new_keep
                    # We could break here to restart the nullspace search with the new matrix,
                    # but that's expensive. Let's just try to find more in the same pass?
                    # Actually, the nullspace was computed for the OLD matrix.
                    # So we should break and recompute.
                    break
            
            if r_next < r:
                # Recurse or loop to try further reduction
                return try_reduce(U_next, V_next, W_next, r_next)
            
        except:
            pass
            
        return U_cur, V_cur, W_cur, r

    U_final, V_final, W_final, r_final = try_reduce(U, V, W, 66)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
