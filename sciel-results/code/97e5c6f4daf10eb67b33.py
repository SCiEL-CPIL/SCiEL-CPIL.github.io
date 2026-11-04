"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def strassen_decomp():
        """Build rank-44 decomposition using Strassen for 2x2x2 and schoolbook for 1x2x2 blocks."""
        rank = 44
        terms = []

        def add_term(u_e, v_e, w_e):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            for idx, val in u_e:
                u_vec[idx] = val
            for idx, val in v_e:
                v_vec[idx] = val
            for idx, val in w_e:
                w_vec[idx] = val
            terms.append((u_vec, v_vec, w_vec))

        def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
            """
            Adds 7 terms for 2x2x2 Strassen decomposition.
            A: [[a,b],[c,d]], B: [[e,f],[g,h]], C: [[w11,w12],[w21,w22]]
            """
            # M1 = (a+d)(e+h)
            add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
            # M2 = (c+d)e
            add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
            # M3 = a(f-h)
            add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
            # M4 = d(g-e)
            add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
            # M5 = (a+b)h
            add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
            # M6 = (c-a)(e+f)
            add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
            # M7 = (b-d)(g+h)
            add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])

        def add_rank4(a, b, c, d, e, f, w1, w2):
            """
            Adds 4 terms for 1x2 x 2x2 -> 1x2.
            A row: [a,b], B: [[c,d],[e,f]], C row: [w1,w2]
            """
            add_term([(a,1)], [(c,1)], [(w1,1)])
            add_term([(b,1)], [(e,1)], [(w1,1)])
            add_term([(a,1)], [(d,1)], [(w2,1)])
            add_term([(b,1)], [(f,1)], [(w2,1)])

        # A(3x4) idx: i*4+j
        # B(4x4) idx: j*4+k
        # W(4x3) idx: k*3+i

        # Block structure: rows of A split (2,1), cols of A split (2,2)
        # rows of B split (2,2), cols of B split (2,2)
        # C = A*B, C is 3x4, W idx k*3+i

        # Case 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)

        # Case 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)

        # Case 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)

        # Case 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)

        # Case 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
        add_rank4(8, 9, 0, 1, 4, 5, 2, 5)

        # Case 6: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
        add_rank4(8, 9, 2, 3, 6, 7, 8, 11)

        # Case 7: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
        add_rank4(10, 11, 8, 9, 12, 13, 2, 5)

        # Case 8: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
        add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

        # Assemble matrices
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for r in range(rank):
            u_vec, v_vec, w_vec = terms[r]
            for i in range(n * m):
                U[i, r] = u_vec[i]
            for j in range(m * p):
                V[j, r] = v_vec[j]
            for k in range(p * n):
                W[k, r] = w_vec[k]

        return U, V, W, rank

    def try_alternative_split():
        """
        Try a different block split to see if we can get lower rank.
        Split A rows as (1,2) instead of (2,1).
        A(3x4): rows (1,2), cols (2,2)
        B(4x4): rows (2,2), cols (2,2)
        """
        rank = 44
        terms = []

        def add_term(u_e, v_e, w_e):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            for idx, val in u_e:
                u_vec[idx] = val
            for idx, val in v_e:
                v_vec[idx] = val
            for idx, val in w_e:
                w_vec[idx] = val
            terms.append((u_vec, v_vec, w_vec))

        def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
            add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
            add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
            add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
            add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
            add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
            add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
            add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])

        def add_rank4(a, b, c, d, e, f, w1, w2):
            add_term([(a,1)], [(c,1)], [(w1,1)])
            add_term([(b,1)], [(e,1)], [(w1,1)])
            add_term([(a,1)], [(d,1)], [(w2,1)])
            add_term([(b,1)], [(f,1)], [(w2,1)])

        # A rows: R0=0, R1=1, R2=2; A cols: C0=0, C1=1, C2=2, C3=3
        # B rows: R0=0, R1=1, R2=2, R3=3; B cols: C0=0, C1=1, C2=2, C3=3
        # W: k*3+i

        # Case 1: A[0:1, 0:2] * B[0:2, 0:2] -> C[0:1, 0:2]  (1x2 x 2x2 -> 1x2, rank 4)
        add_rank4(0, 1, 0, 1, 4, 5, 0, 3)

        # Case 2: A[0:1, 0:2] * B[0:2, 2:4] -> C[0:1, 2:4]  (1x2 x 2x2 -> 1x2, rank 4)
        add_rank4(0, 1, 2, 3, 6, 7, 6, 9)

        # Case 3: A[0:1, 2:4] * B[2:4, 0:2] -> C[0:1, 0:2]  (1x2 x 2x2 -> 1x2, rank 4)
        add_rank4(2, 3, 8, 9, 12, 13, 0, 3)

        # Case 4: A[0:1, 2:4] * B[2:4, 2:4] -> C[0:1, 2:4]  (1x2 x 2x2 -> 1x2, rank 4)
        add_rank4(2, 3, 10, 11, 14, 15, 6, 9)

        # Case 5: A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]  (2x2 x 2x2 -> 2x2, rank 7)
        add_strassen(4, 5, 8, 9, 0, 1, 4, 5, 3, 6, 4, 7)

        # Case 6: A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]  (2x2 x 2x2 -> 2x2, rank 7)
        add_strassen(4, 5, 8, 9, 2, 3, 6, 7, 9, 12, 10, 13)

        # Case 7: A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]  (2x2 x 2x2 -> 2x2, rank 7)
        add_strassen(6, 7, 10, 11, 8, 9, 12, 13, 3, 6, 4, 7)

        # Case 8: A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]  (2x2 x 2x2 -> 2x2, rank 7)
        add_strassen(6, 7, 10, 11, 10, 11, 14, 15, 9, 12, 10, 13)

        if len(terms) != rank:
            return None

        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for r in range(rank):
            u_vec, v_vec, w_vec = terms[r]
            for i in range(n * m):
                U[i, r] = u_vec[i]
            for j in range(m * p):
                V[j, r] = v_vec[j]
            for k in range(p * n):
                W[k, r] = w_vec[k]

        return U, V, W, rank

    # Try the main decomposition first
    U_str, V_str, W_str, rank_str = strassen_decomp()
    if verify_decomposition(U_str, V_str, W_str, rank_str):
        best_U, best_V, best_W = U_str, V_str, W_str
        best_rank = rank_str

    # Try alternative split
    alt = try_alternative_split()
    if alt is not None:
        U_alt, V_alt, W_alt, rank_alt = alt
        if verify_decomposition(U_alt, V_alt, W_alt, rank_alt):
            if rank_alt < best_rank:
                best_U, best_V, best_W = U_alt, V_alt, W_alt
                best_rank = rank_alt

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
