import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 6

    # Strategy: 4x4x4 via Kronecker Strassen (rank 49) + 4x4x2 via 4x Strassen (rank 28)
    # Total rank: 77

    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77

    col = 0

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        return c * n + r  # W[k*n+i] where i=row, k=col

    def add_term(u_entries, v_entries, w_entries, c):
        for idx, val in u_entries:
            U[idx][c] += val
        for idx, val in v_entries:
            V[idx][c] += val
        for idx, val in w_entries:
            W[idx][c] += val
        return c + 1

    # Strassen for 2x2 * 2x2
    def get_strassen_terms():
        """Return list of (u_entries, v_entries, w_entries) for Strassen 2x2x2."""
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]))
        # M2 = (A21+A22)B11
        terms.append(([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]))
        # M3 = A11(B12-B22)
        terms.append(([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]))
        # M4 = A22(B21-B11)
        terms.append(([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]))
        # M5 = (A11+A12)B22
        terms.append(([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]))
        # M6 = (A21-A11)(B11+B12)
        terms.append(([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]))
        # M7 = (A12-A22)(B21+B22)
        terms.append(([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]))
        return terms

    strassen_terms = get_strassen_terms()

    # Part 1: 4x4x4 via Kronecker product of two Strassen decompositions
    for i in range(7):
        for j in range(7):
            u_i, v_i, w_i = strassen_terms[i]
            u_j, v_j, w_j = strassen_terms[j]

            u_entries = []
            for (idx1, val1) in u_i:
                for (idx2, val2) in u_j:
                    r1, c1 = idx1 // 2, idx1 % 2
                    r2, c2 = idx2 // 2, idx2 % 2
                    a_r = r1 * 2 + r2
                    a_c = c1 * 2 + c2
                    u_entries.append((a_idx(a_r, a_c), val1 * val2))

            v_entries = []
            for (idx1, val1) in v_i:
                for (idx2, val2) in v_j:
                    r1, c1 = idx1 // 2, idx1 % 2
                    r2, c2 = idx2 // 2, idx2 % 2
                    b_r = r1 * 2 + r2
                    b_c = c1 * 2 + c2
                    v_entries.append((b_idx(b_r, b_c), val1 * val2))

            w_entries = []
            for (idx1, val1) in w_i:
                for (idx2, val2) in w_j:
                    r1, c1 = idx1 // 2, idx1 % 2
                    r2, c2 = idx2 // 2, idx2 % 2
                    c_r = r1 * 2 + r2
                    c_c = c1 * 2 + c2
                    w_entries.append((c_idx(c_r, c_c), val1 * val2))

            col = add_term(u_entries, v_entries, w_entries, col)

    # Part 2: 4x4x2 (B columns 4-5) via 4 Strassen decompositions
    def add_strassen_2x2x2_offset(ar0, ac0, br0, bc0, cr0, cc0, c):
        for (u_entries, v_entries, w_entries) in strassen_terms:
            u_off = [(a_idx(ar0 + r, ac0 + c), v) for (idx, v) in u_entries for (r, c) in [(idx // 2, idx % 2)]]
            v_off = [(b_idx(br0 + r, bc0 + c), v) for (idx, v) in v_entries for (r, c) in [(idx // 2, idx % 2)]]
            w_off = [(c_idx(cr0 + r, cc0 + c), v) for (idx, v) in w_entries for (r, c) in [(idx // 2, idx % 2)]]
            c = add_term(u_off, v_off, w_off, c)
        return c

    col = add_strassen_2x2x2_offset(0, 0, 0, 4, 0, 4, col)
    col = add_strassen_2x2x2_offset(0, 2, 2, 4, 0, 4, col)
    col = add_strassen_2x2x2_offset(2, 0, 0, 4, 2, 4, col)
    col = add_strassen_2x2x2_offset(2, 2, 2, 4, 2, 4, col)

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify with multiple random matrices for robustness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify_decomposition():
        np.random.seed(42)
        for trial in range(3):
            A_test = np.random.randn(4, 4)
            B_test = np.random.randn(4, 6)
            C_expected = A_test @ B_test

            C_computed = np.zeros((4, 6))
            for r in range(rank):
                alpha_r = np.dot(U_np[:, r], A_test.flatten())
                beta_r = np.dot(V_np[:, r], B_test.flatten())
                C_computed += alpha_r * beta_r * W_np[:, r].reshape(6, 4).T

            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-10:
                return False
        return True

    if not verify_decomposition():
        # Fall back to naive
        rank_naive = n * m * p
        U = [[0] * rank_naive for _ in range(n * m)]
        V = [[0] * rank_naive for _ in range(m * p)]
        W = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
