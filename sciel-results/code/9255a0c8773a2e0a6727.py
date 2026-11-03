import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 8

    # Schoolbook fallback (rank 64)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strategy: Partition into 8 independent 2x2 * 2x2 sub-problems
    # A is 2x4: partition columns into 2 groups of 2: A[:, 0:2] and A[:, 2:4]
    # B is 4x8: partition rows into 2 groups of 2 matching A's columns,
    #           and partition columns into 4 groups of 2
    # So we have 2 (j-groups) * 4 (k-groups) = 8 sub-products of size 2x2 * 2x2
    # Each 2x2 * 2x2 can be done with Strassen rank 7
    # Total rank = 8 * 7 = 56

    # Strassen 2x2 * 2x2 decomposition:
    # C = A * B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # For each sub-problem s (jg in 0..1, kg in 0..3):
    # A_sub = A[:, 2*jg : 2*jg+2]  -> indices in A: i*m + (2*jg + j_loc) for i in 0..1, j_loc in 0..1
    # B_sub = B[2*jg : 2*jg+2, 2*kg : 2*kg+2] -> indices in B: (2*jg + j_loc)*p + (2*kg + k_loc)
    # C_sub = C_sub[2*kg : 2*kg+2, :] where output row = 2*kg + k_loc, col = i
    # Output index: k*n + i = (2*kg + k_loc)*2 + i

    U_list = []
    V_list = []
    W_list = []

    # Strassen factors for 2x2*2x2:
    # We define 7 terms. For each term, we specify:
    # - coefficients for A elements (a,b,c,d) -> U entries
    # - coefficients for B elements (e,f,g,h) -> V entries
    # - coefficients for C elements (c00,c01,c10,c11) -> W entries

    # A elements: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B elements: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C elements: c00=C[0,0], c01=C[0,1], c10=C[1,0], c11=C[1,1]

    # Term 1: M1 = (a+d)(e+h)
    # U: a:1, d:1; V: e:1, h:1; W: c00:1, c11:1
    # Term 2: M2 = (c+d)e
    # U: c:1, d:1; V: e:1; W: c10:1, c11:-1
    # Term 3: M3 = a(f-h)
    # U: a:1; V: f:1, h:-1; W: c01:1, c11:1
    # Term 4: M4 = d(g-e)
    # U: d:1; V: g:1, e:-1; W: c00:1, c10:1
    # Term 5: M5 = (a+b)h
    # U: a:1, b:1; V: h:1; W: c00:-1, c01:1
    # Term 6: M6 = (c-a)(e+f)
    # U: c:1, a:-1; V: e:1, f:1; W: c11:1
    # Term 7: M7 = (b-d)(g+h)
    # U: b:1, d:-1; V: g:1, h:1; W: c00:1

    # For each term, U coefficients for (a,b,c,d), V for (e,f,g,h), W for (c00,c01,c10,c11)
    strassen_terms = [
        # (U_coeffs for a,b,c,d), (V_coeffs for e,f,g,h), (W_coeffs for c00,c01,c10,c11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]

    for jg in range(2):  # 2 groups of j (columns of A / rows of B)
        for kg in range(4):  # 4 groups of k (columns of B)
            for term_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[term_idx]

                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)

                # Fill U: A elements a,b,c,d correspond to A[i, 2*jg + j_loc]
                # A index: i * m + (2*jg + j_loc)
                # a: i=0, j_loc=0 -> 0*4 + 2*jg
                # b: i=0, j_loc=1 -> 0*4 + 2*jg + 1
                # c: i=1, j_loc=0 -> 1*4 + 2*jg
                # d: i=1, j_loc=1 -> 1*4 + 2*jg + 1
                a_idx = 0 * m + (2 * jg + 0)
                b_idx = 0 * m + (2 * jg + 1)
                c_idx = 1 * m + (2 * jg + 0)
                d_idx = 1 * m + (2 * jg + 1)
                a_b_c_d_indices = [a_idx, b_idx, c_idx, d_idx]
                for loc, coeff in enumerate(u_coeffs):
                    if coeff != 0:
                        u_row[a_b_c_d_indices[loc]] = coeff

                # Fill V: B elements e,f,g,h correspond to B[2*jg + j_loc, 2*kg + k_loc]
                # B index: (2*jg + j_loc) * p + (2*kg + k_loc)
                # e: j_loc=0, k_loc=0 -> (2*jg)*8 + 2*kg
                # f: j_loc=0, k_loc=1 -> (2*jg)*8 + 2*kg + 1
                # g: j_loc=1, k_loc=0 -> (2*jg+1)*8 + 2*kg
                # h: j_loc=1, k_loc=1 -> (2*jg+1)*8 + 2*kg + 1
                e_idx = (2 * jg + 0) * p + (2 * kg + 0)
                f_idx = (2 * jg + 0) * p + (2 * kg + 1)
                g_idx = (2 * jg + 1) * p + (2 * kg + 0)
                h_idx = (2 * jg + 1) * p + (2 * kg + 1)
                e_f_g_h_indices = [e_idx, f_idx, g_idx, h_idx]
                for loc, coeff in enumerate(v_coeffs):
                    if coeff != 0:
                        v_row[e_f_g_h_indices[loc]] = coeff

                # Fill W: C elements c00,c01,c10,c11 correspond to C[k_loc, i]
                # Output index: (2*kg + k_loc) * n + i
                # c00: k_loc=0, i=0 -> (2*kg)*2 + 0
                # c01: k_loc=0, i=1 -> (2*kg)*2 + 1
                # c10: k_loc=1, i=0 -> (2*kg+1)*2 + 0
                # c11: k_loc=1, i=1 -> (2*kg+1)*2 + 1
                c00_idx = (2 * kg + 0) * n + 0
                c01_idx = (2 * kg + 0) * n + 1
                c10_idx = (2 * kg + 1) * n + 0
                c11_idx = (2 * kg + 1) * n + 1
                c00_c01_c10_c11_indices = [c00_idx, c01_idx, c10_idx, c11_idx]
                for loc, coeff in enumerate(w_coeffs):
                    if coeff != 0:
                        w_row[c00_c01_c10_c11_indices[loc]] = coeff

                U_list.append(u_row)
                V_list.append(v_row)
                W_list.append(w_row)

    rank = len(U_list)
    # rank should be 56

    # Verify correctness with a random test
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test

        # Compute C via tensor decomposition
        C_computed = np.zeros((p, n))
        U_arr = np.array(U_list).T  # shape (rank, n*m)
        V_arr = np.array(V_list).T  # shape (rank, m*p)
        W_arr = np.array(W_list).T  # shape (rank, p*n)

        # T = sum_r U_r V_r W_r
        # C_flat[k*n+i] = sum_{i,j} A[i*m+j] * B[j*p+k]
        # In tensor form: sum_r (sum_{i,j} U[r, i*m+j] * A[i*m+j]) * (sum_{j,k} V[r, j*p+k] * B[j*p+k]) * W[r, k*n+i]
        # This is not quite right. The tensor decomposition is:
        # T[i*m+j, j*p+k, k*n+i] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        # So C[k*n+i] = sum_{i,j} A[i*m+j] * B[j*p+k] * T[i*m+j, j*p+k, k*n+i]
        #             = sum_r sum_{i,j} A[i*m+j] * B[j*p+k] * U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        #             = sum_r (sum_{i,j} A[i*m+j] * U[r, i*m+j]) * (sum_{j,k} B[j*p+k] * V[r, j*p+k]) * W[r, k*n+i]
        # Wait, that's not right either. Let me re-derive.
        # The multiplication tensor satisfies:
        # C[k*n+i] = sum_{j} A[i*m+j] * B[j*p+k]
        # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
        # A rank-r decomposition: T = sum_r U_r \otimes V_r \otimes W_r
        # means T[a,b,c] = sum_r U[r,a] * V[r,b] * W[r,c]
        # So: sum_{j} A[i*m+j] * B[j*p+k] = sum_r sum_{i',j'} U[r, i'*m+j'] * V[r, j'*p+k] * W[r, k*n+i'] * (delta stuff)
        # Actually, the standard formulation is:
        # C = sum_r (U_r^T A_flat) * (V_r^T B_flat) * W_r^T
        # where A_flat is A reshaped to (n*m,), B_flat to (m*p,), C_flat to (p*n,)
        # C_flat[k*n+i] = sum_r (sum_{a} U[r,a] * A_flat[a]) * (sum_{b} V[r,b] * B_flat[b]) * W[r, k*n+i]

        A_flat = A_test.flatten()  # shape (n*m,)
        B_flat = B_test.flatten()  # shape (m*p,)

        # U_arr is (rank, n*m), V_arr is (rank, m*p), W_arr is (rank, p*n)
        # For each r: scalar_r = sum_a U[r,a]*A_flat[a] * sum_b V[r,b]*B_flat[b]
        # C_flat = W_arr.T @ (scalar_r for each r)
        # i.e., C_flat = (W_arr.T) @ (U_arr @ A_flat) * (V_arr @ B_flat)  -- no, element-wise product of the two vectors, then matrix-vector with W

        uA = U_arr @ A_flat  # shape (rank,)
        vB = V_arr @ B_flat  # shape (rank,)
        C_flat_computed = W_arr.T @ (uA * vB)  # shape (p*n,)

        C_computed = C_flat_computed.reshape((p, n))

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_list, "V": V_list, "W": W_list}
