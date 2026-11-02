import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p  # 52
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Correct Strassen terms for 2x2 x 2x2 = 2x2
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices for columns k0,k1: 0=B[0,k0], 1=B[1,k0], 2=B[0,k1], 3=B[1,k1]
    # W indices: 0=C[0,k0], 1=C[1,k0], 2=C[0,k1], 3=C[1,k1]
    #
    # Standard Strassen:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = a(h-f)
    # M4 = d(g-e)
    # M5 = (a+d)(e+h)
    # M6 = (b+c)(g-e)
    # M7 = (b+d)(f-h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M3
    # C[0,1] = M6
    # C[1,1] = M1 - M3 + M5 + M6
    #
    # Wait, let me re-derive this carefully.
    # Standard Strassen for A*B where A, B, C are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e+b*g, a*f+b*h],[c*e+d*g, c*f+d*h]]
    #
    # Strassen formulas:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = (c+d)h
    # M4 = a(f-h)
    # M5 = d(e-g)
    # M6 = (a+b)h
    # M7 = (c-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M4 + M6
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M2 + M3 + M7
    #
    # Let me verify:
    # C[0,0] = a*e+b*g
    # M1 - M4 + M5 - M7 = (a+b)(e+f) - a(f-h) + d(e-g) - (c-d)(g+h)
    # = ae+af+be+bf - af+ah + de-dg - (cg+ch-dg-dh)
    # = ae+af+be+bf - af+ah + de-dg - cg-ch+dg+dh
    # = ae+be+ah+de-dg-cg-ch+dg+dh  [af-af cancels]
    # = ae+be+ah+de-cg-ch+dh  [dg-dg cancels]
    # Hmm, this doesn't seem right. Let me try the standard version.
    #
    # Actually, the standard Strassen is:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = (c+d)h
    # M4 = a(f-h)
    # M5 = d(e-g)
    # M6 = (a+b)h
    # M7 = (c-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M4 + M6
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M2 + M3 + M7

    strassen_terms = [
        # M1 = (a+b)(e+f)
        ([1, 1, 0, 0], {0: 1, 1: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ([0, 0, 1, 1], {0: 1}, {1: 1}),
        # M3 = (c+d)h
        ([0, 0, 1, 1], {3: 1}, {3: 1}),
        # M4 = a(f-h)
        ([1, 0, 0, 0], {1: 1, 3: -1}, {0: -1, 2: 1}),
        # M5 = d(e-g)
        ([0, 0, 0, 1], {0: 1, 2: -1}, {0: 1}),
        # M6 = (a+b)h
        ([1, 1, 0, 0], {3: 1}, {2: 1}),
        # M7 = (c-d)(g+h)
        ([0, 0, 1, -1], {2: 1, 3: 1}, {0: -1, 1: 1}),
    ]

    # Group columns into blocks of 2 and 1
    # 13 = 6*2 + 1
    blocks = []
    k = 0
    while k < p:
        if k + 1 < p:
            blocks.append((k, k + 1))
            k += 2
        else:
            blocks.append((k,))
            k += 1

    # Calculate total rank
    total_rank = 0
    for block in blocks:
        if len(block) == 2:
            total_rank += 7
        else:
            total_rank += 4

    if total_rank < best_rank:
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for block in blocks:
            if len(block) == 2:
                k0, k1 = block
                # Strassen for columns k0, k1
                # V indices: 0=B[0,k0], 1=B[1,k0], 2=B[0,k1], 3=B[1,k1]
                # W indices: 0=C[0,k0], 1=C[1,k0], 2=C[0,k1], 3=C[1,k1]
                for U_vec, V_dict, W_dict in strassen_terms:
                    for i in range(4):
                        if U_vec[i] != 0:
                            U[i][col] = U_vec[i]
                    for j_idx, val in V_dict.items():
                        # Map local V index to global
                        if j_idx == 0:
                            V[0 * p + k0][col] = val
                        elif j_idx == 1:
                            V[1 * p + k0][col] = val
                        elif j_idx == 2:
                            V[0 * p + k1][col] = val
                        elif j_idx == 3:
                            V[1 * p + k1][col] = val
                    for k_idx, val in W_dict.items():
                        # Map local W index to global
                        if k_idx == 0:
                            W[k0 * n + 0][col] = val
                        elif k_idx == 1:
                            W[k0 * n + 1][col] = val
                        elif k_idx == 2:
                            W[k1 * n + 0][col] = val
                        elif k_idx == 3:
                            W[k1 * n + 1][col] = val
                    col += 1
            else:
                k0 = block[0]
                # Schoolbook for single column k0
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k0][col] = 1
                        W[k0 * n + i][col] = 1
                        col += 1

        # Verify using vectorized approach
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Verify with multiple random tests
        verified = True
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test

            # Build C_decomp efficiently
            C_decomp = np.zeros((n, p))
            for t in range(total_rank):
                U_t = U_np[:, t].reshape(n, m)  # n x m
                V_t = V_np[:, t].reshape(m, p)  # m x p
                W_t = W_np[:, t].reshape(p, n)  # p x n, W_t[k,i]
                # C[i,k] += sum_j U_t[i,j] * V_t[j,k] * W_t[k,i]
                C_decomp += np.einsum('ij,jk,ki->ik', U_t, V_t, W_t)

            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                verified = False
                break

        if verified:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = total_rank

        if time.time() - start_time > deadline - 5:
            return best_result

    return best_result
