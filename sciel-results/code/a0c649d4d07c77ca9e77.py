"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Total standard terms
    total_terms = n * m * p

    # Strategy: Apply Strassen to two disjoint 2x2x2 sub-blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {4,5}
    # Each block saves 1 rank (8 terms replaced by 7), so total rank = 60 - 2 = 58

    rank = total_terms - 2

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the two Strassen blocks
    strassen_block1 = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
    strassen_block2 = {(0,2,4), (0,2,5), (0,3,4), (0,3,5), (1,2,4), (1,2,5), (1,3,4), (1,3,5)}
    all_strassen = strassen_block1 | strassen_block2

    col = 0

    # Add all standard terms except those in the Strassen blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in all_strassen:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # Helper to add a Strassen term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1

    # Strassen formulas for C = A*B where A is 2x2, B is 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # u(i,j) = i*m + j, v(j,k) = j*p + k, w(k,i) = k*n + i
    u00_1, u01_1 = 0*m+0, 0*m+1
    u10_1, u11_1 = 1*m+0, 1*m+1
    v00_1, v01_1 = 0*p+0, 0*p+1
    v10_1, v11_1 = 1*p+0, 1*p+1
    w00_1, w01_1 = 0*n+0, 1*n+0
    w10_1, w11_1 = 0*n+1, 1*n+1

    # M1
    add_term([(u00_1, 1), (u11_1, 1)], [(v00_1, 1), (v11_1, 1)], [(w00_1, 1), (w11_1, 1)])
    # M2
    add_term([(u10_1, 1), (u11_1, 1)], [(v00_1, 1)], [(w10_1, 1), (w11_1, -1)])
    # M3
    add_term([(u00_1, 1)], [(v01_1, 1), (v11_1, -1)], [(w01_1, 1), (w11_1, 1)])
    # M4
    add_term([(u11_1, 1)], [(v10_1, 1), (v00_1, -1)], [(w00_1, 1), (w10_1, 1)])
    # M5
    add_term([(u00_1, 1), (u01_1, 1)], [(v11_1, 1)], [(w00_1, -1), (w01_1, 1)])
    # M6
    add_term([(u10_1, 1), (u00_1, -1)], [(v00_1, 1), (v01_1, 1)], [(w11_1, 1)])
    # M7
    add_term([(u01_1, 1), (u11_1, -1)], [(v10_1, 1), (v11_1, 1)], [(w00_1, 1)])

    # Block 2: i in {0,1}, j in {2,3}, k in {4,5}
    u00_2, u01_2 = 0*m+2, 0*m+3
    u10_2, u11_2 = 1*m+2, 1*m+3
    v00_2, v01_2 = 2*p+4, 2*p+5
    v10_2, v11_2 = 3*p+4, 3*p+5
    w00_2, w01_2 = 4*n+0, 5*n+0
    w10_2, w11_2 = 4*n+1, 5*n+1

    # M1
    add_term([(u00_2, 1), (u11_2, 1)], [(v00_2, 1), (v11_2, 1)], [(w00_2, 1), (w11_2, 1)])
    # M2
    add_term([(u10_2, 1), (u11_2, 1)], [(v00_2, 1)], [(w10_2, 1), (w11_2, -1)])
    # M3
    add_term([(u00_2, 1)], [(v01_2, 1), (v11_2, -1)], [(w01_2, 1), (w11_2, 1)])
    # M4
    add_term([(u11_2, 1)], [(v10_2, 1), (v00_2, -1)], [(w00_2, 1), (w10_2, 1)])
    # M5
    add_term([(u00_2, 1), (u01_2, 1)], [(v11_2, 1)], [(w00_2, -1), (w01_2, 1)])
    # M6
    add_term([(u10_2, 1), (u00_2, -1)], [(v00_2, 1), (v01_2, 1)], [(w11_2, 1)])
    # M7
    add_term([(u01_2, 1), (u11_2, -1)], [(v10_2, 1), (v11_2, 1)], [(w00_2, 1)])

    # Verify col is correct
    assert col == rank, f"Expected col={rank}, got {col}"

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: single Strassen block, rank 59
        rank_fallback = total_terms - 1
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]

        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_block1:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_f[u_idx][col_f] = 1
                    V_f[v_idx][col_f] = 1
                    W_f[w_idx][col_f] = 1
                    col_f += 1

        def add_term_f(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_f
            for u_idx, coeff in u_coeffs:
                U_f[u_idx][col_f] += coeff
            for v_idx, coeff in v_coeffs:
                V_f[v_idx][col_f] += coeff
            for w_idx, coeff in w_coeffs:
                W_f[w_idx][col_f] += coeff
            col_f += 1

        add_term_f([(u00_1, 1), (u11_1, 1)], [(v00_1, 1), (v11_1, 1)], [(w00_1, 1), (w11_1, 1)])
        add_term_f([(u10_1, 1), (u11_1, 1)], [(v00_1, 1)], [(w10_1, 1), (w11_1, -1)])
        add_term_f([(u00_1, 1)], [(v01_1, 1), (v11_1, -1)], [(w01_1, 1), (w11_1, 1)])
        add_term_f([(u11_1, 1)], [(v10_1, 1), (v00_1, -1)], [(w00_1, 1), (w10_1, 1)])
        add_term_f([(u00_1, 1), (u01_1, 1)], [(v11_1, 1)], [(w00_1, -1), (w01_1, 1)])
        add_term_f([(u10_1, 1), (u00_1, -1)], [(v00_1, 1), (v01_1, 1)], [(w11_1, 1)])
        add_term_f([(u01_1, 1), (u11_1, -1)], [(v10_1, 1), (v11_1, 1)], [(w00_1, 1)])

        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
