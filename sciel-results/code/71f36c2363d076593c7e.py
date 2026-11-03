"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, plus a column-elimination pass
to reduce rank below the naive combination.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 0:2]: direct
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 2:4]: direct
    for i in range(2):
        for k in range(2, 4):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 4:6]: direct
    for i in range(2):
        for k in range(4, 6):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 6:8]: direct
    for i in range(2):
        for k in range(6, 8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:2]: direct
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 2:4]: direct
    for j in range(2):
        for k in range(2, 4):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 4:6]: direct
    for j in range(2):
        for k in range(4, 6):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 6:8]: direct
    for j in range(2):
        for k in range(6, 8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:2]: direct
    for k in range(2):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 2:4]: direct
    for k in range(2, 4):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 4:6]: direct
    for k in range(4, 6):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 6:8]: direct
    for k in range(6, 8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank by finding linearly dependent columns
    # A column c is removable if its contribution can be expressed as a linear combination of others
    # We check: for each column c, can we find coefficients alpha such that
    # U[:,c] = sum_a alpha_a * U[:,a], V[:,c] = sum_a alpha_a * V[:,a], W[:,c] = sum_a alpha_a * W[:,a]
    # This is a linear system. We try to eliminate columns one by one.
    
    def try_remove_column(Ua, Va, Wa, col_idx, max_iter=200):
        """Try to remove column col_idx by expressing it as combination of others."""
        r = Ua.shape[1]
        if r <= 1:
            return False, Ua, Va, Wa
        
        # Build the linear system: find alpha such that
        # U[:,col] = U[:,others] @ alpha
        # V[:,col] = V[:,others] @ alpha  
        # W[:,col] = W[:,others] @ alpha
        
        others = [i for i in range(r) if i != col_idx]
        U_others = Ua[:, others]
        V_others = Va[:, others]
        W_others = Wa[:, others]
        
        U_target = Ua[:, col_idx]
        V_target = Va[:, col_idx]
        W_target = Wa[:, col_idx]
        
        # Stack all three matrices
        A_mat = np.vstack([U_others, V_others, W_others])  # (3*r, r-1)
        b_vec = np.concatenate([U_target, V_target, W_target])  # (3*r,)
        
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return False, Ua, Va, Wa
        
        # Check if the solution is exact
        recon = A_mat @ alpha
        if np.max(np.abs(recon - b_vec)) < 1e-8:
            # Remove the column
            U_new = np.delete(Ua, col_idx, axis=1)
            V_new = np.delete(Va, col_idx, axis=1)
            W_new = np.delete(Wa, col_idx, axis=1)
            return True, U_new, V_new, W_new
        
        return False, Ua, Va, Wa
    
    # Iteratively try to remove columns
    improved = True
    while improved:
        if time.time() > DEADLINE - 5:
            break
        improved = False
        r = U_arr.shape[1]
        for c in range(r):
            if time.time() > DEADLINE - 5:
                break
            removable, U_arr, V_arr, W_arr = try_remove_column(U_arr, V_arr, W_arr, c)
            if removable:
                improved = True
                break
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to the pre-reduction version
        rank = len(W_t)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j][idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k][idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i][idx] = coeff
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
    
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
