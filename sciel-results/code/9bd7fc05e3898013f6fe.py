import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=20):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n * m) @ Ua)
                y = (B.reshape(m * p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected_T = C_expected.T
                if not np.allclose(C_computed, C_expected_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def schoolbook_decomposition():
        """Full schoolbook: rank 80"""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_4x4x4_decomposition():
        """
        Rank-49 double-Strassen for 4x4x4.
        Returns U (16x49), V (20x49), W (16x49) for the 4x4x4 subproblem.
        U: A[i,j] -> index i*4+j
        V: B[j,k] -> index j*4+k
        W: C[i,k] -> index k*4+i (trace convention W[k*n+i])
        """
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]

        # Inner Strassen 2x2x2 definitions
        # Each: (u_pairs, v_pairs, w_pairs)
        # u_pairs: list of (i, j, coeff) for A indices
        # v_pairs: list of (j, k, coeff) for B indices
        # w_pairs: list of (i, k, coeff) for C indices
        inner_defs = [
            # M1 = (A00+A11)(B00+B11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A10+A11)B00
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = A00(B10-B00)
            ([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
            # M4 = A11(B11-B01)
            ([(1,1,1)], [(1,1,1),(0,1,-1)], [(0,0,1),(1,0,1)]),
            # M5 = (A00+A01)B11
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A10-A00)(B00+B01)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7 = (A01-A11)(B10+B11)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        # Super Strassen 4x4x4 definitions
        # Each: (p_coeffs, q_coeffs, c_contribs)
        # p_coeffs: 4 values for the 4 blocks of A
        # q_coeffs: 4 values for the 4 blocks of B
        # c_contribs: list of (br, bc, sign) for output block placement
        super_defs = [
            ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
            ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
            ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
            ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
            ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
            ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
            ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
        ]

        col = 0
        for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
            for i_idx, (u_pairs, v_pairs, w_pairs) in enumerate(inner_defs):
                for i, j, c in u_pairs:
                    # Map inner (i,j) to outer based on p_bc
                    for b_idx, coeff in enumerate(p_bc):
                        if coeff == 0:
                            continue
                        gi = b_idx // 2 * 2 + i
                        gj = b_idx % 2 * 2 + j
                        U[gi * 4 + gj][col] += c * coeff
                for j, k, c in v_pairs:
                    for b_idx, coeff in enumerate(q_bc):
                        if coeff == 0:
                            continue
                        gj = b_idx // 2 * 2 + j
                        gk = b_idx % 2 * 2 + k
                        V[gj * 4 + gk][col] += c * coeff
                for i, k, c in w_pairs:
                    for br, bc, sign in c_contribs:
                        gi = br * 2 + i
                        gk = bc * 2 + k
                        W[gk * 4 + gi][col] += sign * c
                col += 1

        return U, V, W

    def combined_decomposition():
        """
        Combine rank-49 Strassen for columns 0-3 with rank-16 schoolbook for column 4.
        Total rank: 65
        """
        rank = 65
        U = [[0] * rank for _ in range(n * m)]  # 16 x 65
        V = [[0] * rank for _ in range(m * p)]  # 20 x 65
        W = [[0] * rank for _ in range(p * n)]  # 20 x 65

        # Part 1: Strassen for first 4 columns (rank 49)
        U_s, V_s, W_s = strassen_4x4x4_decomposition()
        for i in range(16):
            for c in range(49):
                if U_s[i][c] != 0:
                    U[i][c] = U_s[i][c]
        for i in range(16):
            for c in range(49):
                if V_s[i][c] != 0:
                    V[i][c] = V_s[i][c]
        for i in range(16):
            for c in range(49):
                if W_s[i][c] != 0:
                    W[i][c] = W_s[i][c]

        # Part 2: Schoolbook for 5th column (rank 16)
        col = 49
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 5 + 4][col] = 1
                W[4 * 4 + i][col] = 1
                col += 1

        return U, V, W

    # Try the combined decomposition (rank 65)
    U, V, W = combined_decomposition()
    ok = verify(U, V, W, seed=42, trials=15)
    if ok:
        return {"U": U, "V": V, "W": W}

    # Fallback: full schoolbook (rank 80)
    U, V, W = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
