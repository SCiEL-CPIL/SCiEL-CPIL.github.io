import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1_term(U_vec, V_vec, W_vec):
        U_cols.append(list(U_vec))
        V_cols.append(list(V_vec))
        W_cols.append(list(W_vec))

    def add_strassen(a_i, a_j, b_i, b_k, c_i, c_k):
        """Add 7 Strassen rank-1 terms for 2x2x2 block multiplication."""
        a_i, a_j, b_i, b_k, c_i, c_k = int(a_i), int(a_j), int(b_i), int(b_k), int(c_i), int(c_k)

        # m1: (a+d)(e+h) -> c00 + c11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_i, b_k)] = 1; V1[v_idx(b_i+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U1, V1, W1)

        # m2: (c+d)*e -> c10 - c11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_i, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_rank1_term(U2, V2, W2)

        # m3: a*(f-h) -> c01 + c11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_i, b_k+1)] = 1; V3[v_idx(b_i+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U3, V3, W3)

        # m4: d*(g-e) -> c00 + c10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_i+1, b_k)] = 1; V4[v_idx(b_i, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_rank1_term(U4, V4, W4)

        # m5: (a+b)*h -> -c00 + c01
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_i+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_rank1_term(U5, V5, W5)

        # m6: (c-a)*(e+f) -> c11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_i, b_k)] = 1; V6[v_idx(b_i, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U6, V6, W6)

        # m7: (b-d)*(g+h) -> c00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_i+1, b_k)] = 1; V7[v_idx(b_i+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook(i, j, k):
        U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
        U_t[u_idx(i, j)] = 1
        V_t[v_idx(j, k)] = 1
        W_t[w_idx(i, k)] = 1
        add_rank1_term(U_t, V_t, W_t)

    # Strategy: partition the 3x3 * 3x7 multiplication into blocks.
    #
    # Row 2 of A (index 2) multiplies all of B: 1x3x7 = 21 schoolbook terms
    # Rows 0,1 of A (2 rows) multiply columns 0,1 of B: 2x2x2 Strassen = 7 terms
    # Rows 0,1 of A multiply column 2 of B: 2x1x2 schoolbook = 4 terms
    # Rows 0,1 of A multiply columns 3,4 of B: 2x2x2 Strassen = 7 terms
    # Rows 0,1 of A multiply column 5 of B: 2x1x2 schoolbook = 4 terms
    # Rows 0,1 of A multiply column 6 of B: 2x1x1 schoolbook = 2 terms
    #
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, let's think of it as:
    # A is 3x3, B is 3x7, C is 3x7
    #
    # Block decomposition by rows of A and columns of B:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]: Strassen 2x2x2 = 7
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]: Strassen 2x2x2 = 7
    # A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]: Schoolbook 2x2x1 = 4
    # A[0:2, 0:2] * B[0:2, 5:7] -> C[0:2, 5:7]: Strassen 2x2x2 = 7
    # A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]: Schoolbook 1x2x7 = 14
    # A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]: Schoolbook 3x1x7 = 21
    #
    # But wait - this double counts! The terms A[0:2,0:2]*B[0:2,...] already cover
    # the contribution from A rows 0,1 and B cols 0,1. The term A[0:3,2]*B[2,...]
    # covers A rows 0,1,2 and B col 2. And A[2,0:2]*B[0:2,...] covers A row 2, B cols 0,1.
    # So the full sum is:
    # C = A[0:2,0:2]*B[0:2,0:7] + A[0:2,2]*B[2,0:7] + A[2,0:2]*B[0:2,0:7] + A[2,2]*B[2,0:7]
    #
    # Let me redo:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]: Strassen = 7
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]: Strassen = 7
    # A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]: Schoolbook 2x2x1 = 4
    # A[0:2, 0:2] * B[0:2, 5:7] -> C[0:2, 5:7]: Strassen = 7
    # A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]: Schoolbook 1x2x7 = 14
    # A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]: Schoolbook 3x1x7 = 21
    #
    # Total: 7+7+4+7+14+21 = 60
    #
    # Can we do better? Let's try:
    # A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]: This is 2x2x4.
    #   Split into two 2x2x2 Strassen: 7+7 = 14 (same as before for cols 0:2 and 2:4)
    #
    # What about A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]: 2x2x3
    #   2x2x3 = 2x2x2 (Strassen=7) + 2x2x1 (schoolbook=4) = 11
    #   Or: 2x2x3 schoolbook = 12. So Strassen+schoolbook saves 1.
    #
    # Alternative partition:
    # A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]: 2x2x4 = two 2x2x2 = 14
    # A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]: 2x2x3 = 7+4 = 11
    # A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]: 1x2x7 = 14
    # A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]: 3x1x7 = 21
    # Total: 14+11+14+21 = 60 (same)
    #
    # Let's try another split of the 2x2x4:
    # Actually 2x2x4 can't be done in less than 14 with known methods (2*2*4=16 schoolbook,
    # two Strassen 2x2x2 = 14).
    #
    # What about using 2x3x7 for the top 2 rows?
    # A[0:2, 0:3] * B[0:3, 0:7] -> C[0:2, 0:7]: 2x3x7
    #   = A[0:2,0:2]*B[0:2,0:7] + A[0:2,2]*B[2,0:7]
    #   = (7+7+4+7) + (2*1*7=14) = 25+14 = 39
    # Then A[2,0:3]*B[0:3,0:7] -> C[2,0:7]: 1x3x7 = 21
    # Total: 39+21 = 60 (same)
    #
    # The current 60 seems to be a natural bound. Let me try to verify it works.

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (Strassen: 7)
    add_strassen(0, 0, 0, 0, 0, 0)

    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (Strassen: 7)
    add_strassen(0, 0, 0, 2, 0, 2)

    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (Schoolbook 2x2x1: 4)
    for i in range(2):
        for j in range(2):
            add_schoolbook(i, j, 4)

    # Block 4: A[0:2,0:2] * B[0:2,5:7] -> C[0:2,5:7]  (Strassen: 7)
    add_strassen(0, 0, 0, 5, 0, 5)

    # Block 5: A[2,0:2] * B[0:2,0:7] -> C[2,0:7]  (Schoolbook 1x2x7: 14)
    for j in range(2):
        for k in range(7):
            add_schoolbook(2, j, k)

    # Block 6: A[0:3,2] * B[2,0:7] -> C[0:3,0:7]  (Schoolbook 3x1x7: 21)
    for i in range(3):
        for k in range(7):
            add_schoolbook(i, 2, k)

    rank = len(U_cols)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for i in range(n * m):
            if U_cols[r][i] != 0:
                U[i][r] = U_cols[r][i]
        for i in range(m * p):
            if V_cols[r][i] != 0:
                V[i][r] = V_cols[r][i]
        for i in range(p * n):
            if W_cols[r][i] != 0:
                W[i][r] = W_cols[r][i]

    # Verification with multiple random tests
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify():
        np.random.seed(12345)
        for trial in range(5):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_true = A_test @ B_test

            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for r in range(rank):
                            val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
                    C_comp[i, k] = val

            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True

    if not verify():
        # Fallback to pure schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
