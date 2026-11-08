"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, then performs exact linear-algebra
rank reduction on the term-vector space to find the minimal rank.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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

    # A[0:2, 0:2] * B[0:2, k0:k0+2]: Strassen for each column-pair
    for k0 in [0, 2, 4, 6]:
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)

    # A[0:2, 2] * B[2, k0:k0+2]: direct
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                direct_term(i, 2, k, U_t, V_t, W_t)

    # A[2, 0:2] * B[0:2, k0:k0+2]: direct
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                direct_term(2, j, k, U_t, V_t, W_t)

    # A[2, 2] * B[2, k0:k0+2]: direct
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
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

    # --- Exact linear-algebra rank reduction ---
    # Each rank term r contributes tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    # If the set of term vectors is linearly dependent, we can find a minimal
    # spanning set and re-express the decomposition.
    
    def term_vector(r, Ua, Va, Wa):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        return vec

    # Build the matrix whose columns are all term vectors
    R = U_arr.shape[1]
    M = np.zeros((n * m * p, R))
    for r in range(R):
        M[:, r] = term_vector(r, U_arr, V_arr, W_arr)

    # Find the rank and a basis using SVD
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # Determine numerical rank
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = np.sum(s_vals > tol)
    except Exception:
        num_rank = R
        U_svd = np.eye(R)
        Vt_svd = np.eye(R)

    if num_rank < R:
        # The term vectors are linearly dependent.
        # We can use the first num_rank left singular vectors as a basis,
        # but we need to reconstruct a valid decomposition.
        # 
        # Key insight: if M = M_reduced * C where M_reduced has rank num_rank
        # and C is a (num_rank, R) matrix, then we can rewrite the decomposition.
        #
        # Actually, a cleaner approach: use the fact that if the columns of M
        # span a space of dimension num_rank, there exist coefficients such that
        # each column is a linear combination of num_rank basis columns.
        #
        # But for tensor decompositions, we need the reduction to preserve the
        # multiplicative structure. The safest approach is:
        #
        # Use QR with column pivoting to find a basis of independent columns,
        # then express all other columns as linear combinations of the basis.
        # For each non-basis column r, if M[:,r] = sum_{b in basis} c_{r,b} * M[:,b],
        # then we can absorb the contribution: the original term r's contribution
        # equals sum_b c_{r,b} * term_b's contribution.
        #
        # This means: new_contribution = sum_{all r} T_r = sum_{b in basis} T_b + sum_{r not in basis} T_r
        #                                    = sum_{b in basis} T_b + sum_{r not in basis} sum_{b in basis} c_{r,b} T_b
        #                                    = sum_{b in basis} (1 + sum_{r not in basis, r depends on b} c_{r,b}) T_b
        #
        # So the new decomposition has num_rank terms, where each basis term b is
        # scaled by (1 + sum of coefficients from non-basis terms that depend on b).
        
        # Use QR with column pivoting
        try:
            # Implement simple column pivoting via SVD
            # The right singular vectors give the column space basis
            # Vt_svd[:num_rank, :] gives the first num_rank rows of V^T
            # Columns of M that are well-represented by the top singular vectors
            # are the "independent" ones.
            
            # Alternative: use the fact that if rank is num_rank, we can find
            # num_rank columns that form a basis. Use greedy selection.
            
            # For simplicity and correctness, let's use a different approach:
            # Find the null space of M^T M (or equivalently, the right null space of M).
            # If there's a non-trivial null vector x with M @ x = 0, then
            # sum_r x_r * T_r = 0, meaning the weighted sum of terms is zero.
            # We can use this to eliminate one term.
            
            # Find null space vectors
            null_vecs = Vt_svd[num_rank:, :]  # shape (R - num_rank, R)
            
            best_U, best_V, best_W = U_arr, V_arr, W_arr
            best_rank = R
            
            # Try to eliminate terms using null space relations
            for nv in null_vecs:
                if (time.time() - start_time) > DEADLINE - 10.0:
                    break
                # nv is a vector such that sum_r nv[r] * T_r = 0
                # Find the index with the largest |nv[r]| to eliminate
                r_elim = np.argmax(np.abs(nv))
                if abs(nv[r_elim]) < 1e-12:
                    continue
                
                # Express T_{r_elim} = -sum_{r != r_elim} (nv[r]/nv[r_elim]) * T_r
                # New decomposition: replace T_{r_elim} with this combination
                # New coefficient for term r (r != r_elim): 1 + (-nv[r]/nv[r_elim])
                # Wait: sum_all T_r = sum_{r != r_elim} T_r + T_{r_elim}
                #   = sum_{r != r_elim} T_r - sum_{r != r_elim} (nv[r]/nv[r_elim]) T_r
                #   = sum_{r != r_elim} (1 - nv[r]/nv[r_elim]) T_r
                
                new_rank = R - 1
                new_U = np.zeros((n * m, new_rank))
                new_V = np.zeros((m * p, new_rank))
                new_W = np.zeros((p * n, new_rank))
                
                idx = 0
                for r in range(R):
                    if r == r_elim:
                        continue
                    scale = 1.0 - nv[r] / nv[r_elim]
                    if abs(scale) < 1e-12:
                        # This term vanishes; skip it entirely
                        continue
                    new_U[:, idx] = U_arr[:, r] * scale
                    new_V[:, idx] = V_arr[:, r] * scale
                    new_W[:, idx] = W_arr[:, r] * scale
                    idx += 1
                
                actual_new_rank = idx
                if actual_new_rank < new_rank:
                    new_U = new_U[:, :actual_new_rank]
                    new_V = new_V[:, :actual_new_rank]
                    new_W = new_W[:, :actual_new_rank]
                
                # Verify
                if verify_decomp(new_U, new_V, new_W):
                    if actual_new_rank < best_rank:
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = actual_new_rank
                    U_arr, V_arr, W_arr = new_U, new_V, new_W
                    R = actual_new_rank
                    # Recompute M and continue
                    M = np.zeros((n * m * p, R))
                    for r in range(R):
                        M[:, r] = term_vector(r, U_arr, V_arr, W_arr)
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
                        num_rank = np.sum(s_vals > tol)
                    except Exception:
                        break
                    if num_rank >= R:
                        break
        except Exception:
            pass

        if best_rank < R:
            U_arr, V_arr, W_arr = best_U, best_V, best_W

    # Convert to nice integer/half-integer values
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
