"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Strategy: Decompose 2x5x6 into 2x1x6 (rank 12) + 2x4x6 (rank 42)
    # 2x1x6: each term is independent, rank = 2*1*6 = 12
    # 2x4x6: use 3 Strassen 2x2x2 blocks on j-pairs {0,1},{2,3} and k-pairs {0,1},{2,3},{4,5}
    #   2x2x6 can be decomposed with rank 21 (3 Strassen instances, saving 3 from 24)
    #   So 2x4x6 = 2 * (2x2x6) = 2 * 21 = 42
    # Total rank = 12 + 42 = 54

    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: 2x1x6 block (j=0, all i, k)
    # For i in {0,1}, j=0, k in {0,...,5}: standard terms
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    # Part 1 done: col should be 12
    assert col == 12, f"Part 1: expected col=12, got {col}"

    # Part 2: 2x4x6 block (j in {1,2,3,4}, i in {0,1}, k in {0,...,5})
    # Split j into two pairs: j_pair1 = {1,2}, j_pair2 = {3,4}
    # For each j_pair, split k into three pairs: k_pair1 = {0,1}, k_pair2 = {2,3}, k_pair3 = {4,5}
    # Each (i_pair={0,1}, j_pair, k_pair) is a 2x2x2 Strassen block with rank 7

    def add_strassen_block(i_base, j_base, k_base, col_offset):
        """Add a 2x2x2 Strassen block starting at column col_offset.
        i_base: 0 or 1 (base for i index, actual i = i_base + 0 or 1)
        j_base: base for j index, actual j = j_base + 0 or 1
        k_base: base for k index, actual k = k_base + 0 or 1
        Returns the new column offset (col_offset + 7).
        """
        # Local indices
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)
        
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
        
        w00 = k_base * n + i_base
        w01 = (k_base + 1) * n + i_base
        w10 = k_base * n + (i_base + 1)
        w11 = (k_base + 1) * n + (i_base + 1)
        
        c = col_offset
        
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        for u_idx, coeff in [(u00, 1), (u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1), (v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1), (w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M2 = (A10 + A11)B00 -> C10 - C11
        for u_idx, coeff in [(u10, 1), (u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w10, 1), (w11, -1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M3 = A00(B01 - B11) -> C01 + C11
        for u_idx, coeff in [(u00, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v01, 1), (v11, -1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w01, 1), (w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M4 = A11(B10 - B00) -> C00 + C10
        for u_idx, coeff in [(u11, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v10, 1), (v00, -1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1), (w10, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M5 = (A00 + A01)B11 -> -C00 + C01
        for u_idx, coeff in [(u00, 1), (u01, 1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, -1), (w01, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        for u_idx, coeff in [(u10, 1), (u00, -1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v00, 1), (v01, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w11, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        for u_idx, coeff in [(u01, 1), (u11, -1)]:
            U[u_idx][c] += coeff
        for v_idx, coeff in [(v10, 1), (v11, 1)]:
            V[v_idx][c] += coeff
        for w_idx, coeff in [(w00, 1)]:
            W[w_idx][c] += coeff
        c += 1
        
        return c

    # 6 Strassen blocks for 2x4x6:
    # j_pair1: j in {1,2}, j_base=1
    # j_pair2: j in {3,4}, j_base=3
    # k_pairs: {0,1} k_base=0, {2,3} k_base=2, {4,5} k_base=4

    # j_pair1, k_pair1: i_base=0, j_base=1, k_base=0
    col = add_strassen_block(0, 1, 0, col)
    # j_pair1, k_pair2: i_base=0, j_base=1, k_base=2
    col = add_strassen_block(0, 1, 2, col)
    # j_pair1, k_pair3: i_base=0, j_base=1, k_base=4
    col = add_strassen_block(0, 1, 4, col)
    # j_pair2, k_pair1: i_base=0, j_base=3, k_base=0
    col = add_strassen_block(0, 3, 0, col)
    # j_pair2, k_pair2: i_base=0, j_base=3, k_base=2
    col = add_strassen_block(0, 3, 2, col)
    # j_pair2, k_pair3: i_base=0, j_base=3, k_base=4
    col = add_strassen_block(0, 3, 4, col)

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
        # Fallback: original two-Strassen-block approach, rank 58
        total_terms = n * m * p
        rank_fallback = total_terms - 2
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]

        strassen_block1 = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
        strassen_block2 = {(0,2,4), (0,2,5), (0,3,4), (0,3,5), (1,2,4), (1,2,5), (1,3,4), (1,3,5)}
        all_strassen = strassen_block1 | strassen_block2

        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in all_strassen:
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

        # Block 1
        u00_1, u01_1 = 0*m+0, 0*m+1
        u10_1, u11_1 = 1*m+0, 1*m+1
        v00_1, v01_1 = 0*p+0, 0*p+1
        v10_1, v11_1 = 1*p+0, 1*p+1
        w00_1, w01_1 = 0*n+0, 1*n+0
        w10_1, w11_1 = 0*n+1, 1*n+1
        add_term_f([(u00_1, 1), (u11_1, 1)], [(v00_1, 1), (v11_1, 1)], [(w00_1, 1), (w11_1, 1)])
        add_term_f([(u10_1, 1), (u11_1, 1)], [(v00_1, 1)], [(w10_1, 1), (w11_1, -1)])
        add_term_f([(u00_1, 1)], [(v01_1, 1), (v11_1, -1)], [(w01_1, 1), (w11_1, 1)])
        add_term_f([(u11_1, 1)], [(v10_1, 1), (v00_1, -1)], [(w00_1, 1), (w10_1, 1)])
        add_term_f([(u00_1, 1), (u01_1, 1)], [(v11_1, 1)], [(w00_1, -1), (w01_1, 1)])
        add_term_f([(u10_1, 1), (u00_1, -1)], [(v00_1, 1), (v01_1, 1)], [(w11_1, 1)])
        add_term_f([(u01_1, 1), (u11_1, -1)], [(v10_1, 1), (v11_1, 1)], [(w00_1, 1)])

        # Block 2
        u00_2, u01_2 = 0*m+2, 0*m+3
        u10_2, u11_2 = 1*m+2, 1*m+3
        v00_2, v01_2 = 2*p+4, 2*p+5
        v10_2, v11_2 = 3*p+4, 3*p+5
        w00_2, w01_2 = 4*n+0, 5*n+0
        w10_2, w11_2 = 4*n+1, 5*n+1
        add_term_f([(u00_2, 1), (u11_2, 1)], [(v00_2, 1), (v11_2, 1)], [(w00_2, 1), (w11_2, 1)])
        add_term_f([(u10_2, 1), (u11_2, 1)], [(v00_2, 1)], [(w10_2, 1), (w11_2, -1)])
        add_term_f([(u00_2, 1)], [(v01_2, 1), (v11_2, -1)], [(w01_2, 1), (w11_2, 1)])
        add_term_f([(u11_2, 1)], [(v10_2, 1), (v00_2, -1)], [(w00_2, 1), (w10_2, 1)])
        add_term_f([(u00_2, 1), (u01_2, 1)], [(v11_2, 1)], [(w00_2, -1), (w01_2, 1)])
        add_term_f([(u10_2, 1), (u00_2, -1)], [(v00_2, 1), (v01_2, 1)], [(w11_2, 1)])
        add_term_f([(u01_2, 1), (u11_2, -1)], [(v10_2, 1), (v11_2, 1)], [(w00_2, 1)])

        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
