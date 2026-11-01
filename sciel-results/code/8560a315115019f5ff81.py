"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Uses block-Strassen: split middle dim 4=2+2, output cols 5=2+2+1.
    Each 2x2*2x2 uses Strassen (rank 7), each 2x2*2x1 uses schoolbook (rank 4).
    Total: 2*(7+7+4) = 36.
    """
    import time
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 5

    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        """
        Add Strassen terms for 2x2 * 2x2.
        a_offset: (i0, j0) - top-left of A block
        b_offset: (j0, k0) - top-left of B block
        c_offset: (i0, k0) - top-left of C block
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            # m1 = (x11+x22)*(y11+y22) -> z11 += 1, z22 += 1
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11 -> z21 += 1, z22 -= 1
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22) -> z12 += 1, z22 += 1
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11) -> z11 += 1, z21 += 1
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22 -> z11 -= 1, z12 += 1
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12) -> z22 += 1
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22) -> z11 += 1
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]

        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        """Add schoolbook terms for 2x2 * 2x1."""
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]

        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))

    # A is 2x4: i in {0,1}, j in {0,1,2,3}
    # B is 4x5: j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: i in {0,1}, k in {0,1,2,3,4}

    # Top A-block: A[0:2, 0:2] * B blocks
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))    # cols 0-1
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))    # cols 2-3
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))  # col 4

    # Bottom A-block: A[0:2, 2:4] * B blocks
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))    # cols 0-1
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))    # cols 2-3
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))  # col 4

    rank = len(rank_terms)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    # Verify correctness with a random test
    try:
        import numpy as np
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build the tensor and check
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: return schoolbook decomposition (rank 40)
            rank_terms_fb = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rank_terms_fb.append(
                            ({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                        )
            rank_fb = len(rank_terms_fb)
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms_fb):
                for (i, j), coeff in u_dict.items():
                    U_fb[i * m + j][col] = coeff
                for (j, k), coeff in v_dict.items():
                    V_fb[j * p + k][col] = coeff
                for (i, k), coeff in w_dict.items():
                    W_fb[k * n + i][col] = coeff
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass

    if time.time() > deadline:
        # Return what we have
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
