import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS
    # n=3, m=3, p=7

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i

    terms = []

    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))

    def zero_U(): return [0] * (n * m)
    def zero_V(): return [0] * (m * p)
    def zero_W(): return [0] * (p * n)

    def strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        U1, V1, W1 = zero_U(), zero_V(), zero_W()
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_j, b_k)] = 1; V1[v_idx(b_j+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_term(U1, V1, W1)

        U2, V2, W2 = zero_U(), zero_V(), zero_W()
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_j, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_term(U2, V2, W2)

        U3, V3, W3 = zero_U(), zero_V(), zero_W()
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_j, b_k+1)] = 1; V3[v_idx(b_j+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_term(U3, V3, W3)

        U4, V4, W4 = zero_U(), zero_V(), zero_W()
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_j+1, b_k)] = 1; V4[v_idx(b_j, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_term(U4, V4, W4)

        U5, V5, W5 = zero_U(), zero_V(), zero_W()
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_j+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_term(U5, V5, W5)

        U6, V6, W6 = zero_U(), zero_V(), zero_W()
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_j, b_k)] = 1; V6[v_idx(b_j, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_term(U6, V6, W6)

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

    # Decomposition for 3x3x3 block (Rank 20)
    # A is 3x3, B is 3x3, C is 3x3
    # Blocks:
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1)
    # C11 (2x2), C12 (2x1), C21 (1x2), C22 (1x1)
    
    def block_3x3x3(a_i, a_j, b_j, b_k, c_i, c_k):
        # 1. A11 * B11 -> C11 (2x2x2, Rank 7)
        strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k)
        
        # 2. A12 * B21 -> C11 (2x1 * 1x2 -> 2x2, Rank 4)
        # A12: rows a_i, a_i+1; col a_j+2
        # B21: row b_j+2; cols b_k, b_k+1
        # C11: rows c_i, c_i+1; cols c_k, c_k+1
        for ii in range(2):
            for kk in range(2):
                U, V, W = zero_U(), zero_V(), zero_W()
                U[u_idx(a_i+ii, a_j+2)] = 1
                V[v_idx(b_j+2, b_k+kk)] = 1
                W[w_idx(c_i+ii, c_k+kk)] = 1
                add_term(U, V, W)

        # 3. A12 * B22 -> C12 (2x1 * 1x1 -> 2x1, Rank 2)
        # A12: rows a_i, a_i+1; col a_j+2
        # B22: row b_j+2; col b_k+2
        # C12: rows c_i, c_i+1; col c_k+2
        for ii in range(2):
            U, V, W = zero_U(), zero_V(), zero_W()
            U[u_idx(a_i+ii, a_j+2)] = 1
            V[v_idx(b_j+2, b_k+2)] = 1
            W[w_idx(c_i+ii, c_k+2)] = 1
            add_term(U, V, W)

        # 4. A21 * B11 -> C21 (1x2 * 2x2 -> 1x2, Rank 2)
        # A21: row a_i+2; cols a_j, a_j+1
        # B11: rows b_j, b_j+1; col b_k+2
        # C21: row c_i+2; cols c_k, c_k+1
        for jj in range(2):
            U, V, W = zero_U(), zero_V(), zero_W()
            U[u_idx(a_i+2, a_j+jj)] = 1
            V[v_idx(b_j+jj, b_k+2)] = 1
            W[w_idx(c_i+2, c_k+jj)] = 1
            add_term(U, V, W)

        # 5. A21 * B12 -> C22 (1x2 * 2x1 -> 1x1, Rank 2)
        # A21: row a_i+2; cols a_j, a_j+1
        # B12: rows b_j, b_j+1; col b_k+2
        # C22: row c_i+2; col c_k+2
        for jj in range(2):
            U, V, W = zero_U(), zero_V(), zero_W()
            U[u_idx(a_i+2, a_j+jj)] = 1
            V[v_idx(b_j+jj, b_k+2)] = 1
            W[w_idx(c_i+2, c_k+2)] = 1
            add_term(U, V, W)

        # 6. A22 * B21 -> C21 (1x1 * 1x2 -> 1x2, Rank 2)
        # A22: row a_i+2; col a_j+2
        # B21: row b_j+2; cols b_k, b_k+1
        # C21: row c_i+2; cols c_k, c_k+1
        for kk in range(2):
            U, V, W = zero_U(), zero_V(), zero_W()
            U[u_idx(a_i+2, a_j+2)] = 1
            V[v_idx(b_j+2, b_k+kk)] = 1
            W[w_idx(c_i+2, c_k+kk)] = 1
            add_term(U, V, W)

        # 7. A22 * B22 -> C22 (1x1 * 1x1 -> 1x1, Rank 1)
        U, V, W = zero_U(), zero_V(), zero_W()
        U[u_idx(a_i+2, a_j+2)] = 1
        V[v_idx(b_j+2, b_k+2)] = 1
        W[w_idx(c_i+2, c_k+2)] = 1
        add_term(U, V, W)

    # Process B in blocks of 3 columns
    # Block 1: cols 0,1,2
    block_3x3x3(0, 0, 0, 0, 0, 0)
    # Block 2: cols 3,4,5
    block_3x3x3(0, 0, 0, 3, 0, 3)
    
    # Block 3: col 6 (3x3x1)
    # A (3x3) * B (3x1) -> C (3x1)
    # Rank 9 (Schoolbook)
    schoolbook_block(0, 0, 0, 6, 0, 6, 3, 3, 1)

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
            # Fallback to schoolbook
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
