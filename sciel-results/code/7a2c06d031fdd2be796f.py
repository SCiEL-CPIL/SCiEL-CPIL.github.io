import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS
    # n=3 rows of A, m=3 cols of A (rows of B), p=7 cols of B (cols of C)

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i  # W[k*n+i]

    terms = []

    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))

    def zero_U(): return [0] * (n * m)
    def zero_V(): return [0] * (m * p)
    def zero_W(): return [0] * (p * n)

    def strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        """
        Strassen for A[a_i:a_i+2, a_j:a_j+2] * B[b_j:b_j+2, b_k:b_k+2] -> C[c_i:c_i+2, c_k:c_k+2]
        """
        # m1 = (a00+a11)(b00+b11) -> c00, c11
        U1, V1, W1 = zero_U(), zero_V(), zero_W()
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_j, b_k)] = 1; V1[v_idx(b_j+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_term(U1, V1, W1)

        # m2 = (a10+a11)b00 -> c10, c11
        U2, V2, W2 = zero_U(), zero_V(), zero_W()
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_j, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_term(U2, V2, W2)

        # m3 = a00(b01-b11) -> c01, c11
        U3, V3, W3 = zero_U(), zero_V(), zero_W()
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_j, b_k+1)] = 1; V3[v_idx(b_j+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_term(U3, V3, W3)

        # m4 = a11(b10-b00) -> c00, c10
        U4, V4, W4 = zero_U(), zero_V(), zero_W()
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_j+1, b_k)] = 1; V4[v_idx(b_j, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_term(U4, V4, W4)

        # m5 = (a00+a01)b11 -> c00, c01
        U5, V5, W5 = zero_U(), zero_V(), zero_W()
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_j+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_term(U5, V5, W5)

        # m6 = (a10-a00)(b00+b01) -> c11
        U6, V6, W6 = zero_U(), zero_V(), zero_W()
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_j, b_k)] = 1; V6[v_idx(b_j, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_term(U6, V6, W6)

        # m7 = (a01-a11)(b10+b11) -> c00
        U7, V7, W7 = zero_U(), zero_V(), zero_W()
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_j+1, b_k)] = 1; V7[v_idx(b_j+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_term(U7, V7, W7)

    def schoolbook_block(a_i, a_j, b_j, b_k, c_i, c_k, di, dj, dk):
        for ii in range(di):
            for jj in range(dj):
                for kk in range(dk):
                    U, V, W = zero_U(), zero_V(), zero_W()
                    U[u_idx(a_i+ii, a_j+jj)] = 1
                    V[v_idx(b_j+jj, b_k+kk)] = 1
                    W[w_idx(c_i+ii, c_k+kk)] = 1
                    add_term(U, V, W)

    # Decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split A into rows [0:2] and [2:3], split B into cols [0:2], [2:4], [4:6], [6:7]
    
    # Process 2-col blocks [0:2], [2:4], [4:6]
    for k_start in [0, 2, 4]:
        # A[0:2,0:2]*B[0:2,k:k+2] -> Strassen: 7 terms
        strassen_2x2(0, 0, 0, k_start, 0, k_start)
        # A[0:2,2]*B[2,k:k+2] -> schoolbook: 2*1*2 = 4 terms
        schoolbook_block(0, 2, 2, k_start, 0, k_start, 2, 1, 2)
        # A[2,0:2]*B[0:2,k:k+2] -> schoolbook: 1*2*2 = 4 terms
        schoolbook_block(2, 0, 0, k_start, 2, k_start, 1, 2, 2)
        # A[2,2]*B[2,k:k+2] -> schoolbook: 1*1*2 = 2 terms
        schoolbook_block(2, 2, 2, k_start, 2, k_start, 1, 1, 2)

    # Process single col block [6:7]
    # A[0:2,0:2]*B[0:2,6:7] -> schoolbook: 2*2*1 = 4 terms
    schoolbook_block(0, 0, 0, 6, 0, 6, 2, 2, 1)
    # A[0:2,2]*B[2,6:7] -> schoolbook: 2*1*1 = 2 terms
    schoolbook_block(0, 2, 2, 6, 0, 6, 2, 1, 1)
    # A[2,0:2]*B[0:2,6:7] -> schoolbook: 1*2*1 = 2 terms
    schoolbook_block(2, 0, 0, 6, 2, 6, 1, 2, 1)
    # A[2,2]*B[2,6:7] -> schoolbook: 1*1*1 = 1 term
    schoolbook_block(2, 2, 2, 6, 2, 6, 1, 1, 1)

    rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][c] = u_vec[i]
        for i in range(m * p):
            if v_vec[i] != 0:
                V[i][c] = v_vec[i]
        for i in range(p * n):
            if w_vec[i] != 0:
                W[i][c] = w_vec[i]

    # Verify correctness
    def verify():
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_true = A_test @ B_test
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_comp[i,k] = s
        return np.allclose(C_comp, C_true, atol=1e-9)

    if time.time() < deadline - safety_margin:
        if not verify():
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
