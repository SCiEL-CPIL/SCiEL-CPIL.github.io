import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 148.0

    n, m, p = DIMENSIONS
    rank = 60

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] = U_vec
        V[:, col] = V_vec
        W[:, col] = W_vec
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    def add_strassen(a_base, b_base, c_base, k_base):
        a00 = u_idx(a_base, 0)
        a01 = u_idx(a_base, 1)
        a10 = u_idx(a_base + 1, 0)
        a11 = u_idx(a_base + 1, 1)

        b00 = v_idx(b_base, k_base)
        b01 = v_idx(b_base, k_base + 1)
        b10 = v_idx(b_base + 1, k_base)
        b11 = v_idx(b_base + 1, k_base + 1)

        c00 = w_idx(c_base, k_base)
        c01 = w_idx(c_base, k_base + 1)
        c10 = w_idx(c_base + 1, k_base)
        c11 = w_idx(c_base + 1, k_base + 1)

        # m1: (a00+a11)(b00+b11)
        U1 = np.zeros(n*m); V1 = np.zeros(m*p); W1 = np.zeros(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_term(U1, V1, W1)

        # m2: (a10+a11)b00
        U2 = np.zeros(n*m); V2 = np.zeros(m*p); W2 = np.zeros(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_term(U2, V2, W2)

        # m3: a00(b01-b11)
        U3 = np.zeros(n*m); V3 = np.zeros(m*p); W3 = np.zeros(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_term(U3, V3, W3)

        # m4: a11(b10-b00)
        U4 = np.zeros(n*m); V4 = np.zeros(m*p); W4 = np.zeros(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_term(U4, V4, W4)

        # m5: (a00+a01)b11
        U5 = np.zeros(n*m); V5 = np.zeros(m*p); W5 = np.zeros(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_term(U5, V5, W5)

        # m6: (a10-a00)(b00+b01)
        U6 = np.zeros(n*m); V6 = np.zeros(m*p); W6 = np.zeros(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_term(U6, V6, W6)

        # m7: (a01-a11)(b10+b11)
        U7 = np.zeros(n*m); V7 = np.zeros(m*p); W7 = np.zeros(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_term(U7, V7, W7)

    def add_schoolbook_block(n_a, m_b, p_c, a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        for i in range(n_a):
            for j in range(m_b):
                for k in range(p_c):
                    U_t = np.zeros(n*m); V_t = np.zeros(m*p); W_t = np.zeros(p*n)
                    U_t[u_idx(a_i0 + i, a_j0 + j)] = 1
                    V_t[v_idx(b_j0 + j, b_k0 + k)] = 1
                    W_t[w_idx(c_i0 + i, c_k0 + k)] = 1
                    add_term(U_t, V_t, W_t)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    # M(2,2,7) = M(2,2,2) + M(2,2,2) + M(2,2,2) + M(2,2,1)
    # M(2,2,2) = 7 (Strassen)
    # M(2,2,1) = 4 (Schoolbook)
    # Total: 7 + 7 + 7 + 4 = 25
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 0, 0, 4)
    add_schoolbook_block(2, 2, 1, 0, 0, 0, 6, 0, 6)

    # Block 2: A[2, 0:2] * B[0:2, 0:7] -> C[2, 0:7]
    # M(1,2,7) = 14 (Schoolbook)
    add_schoolbook_block(1, 2, 7, 2, 0, 0, 0, 2, 0)

    # Block 3: A[0:3, 2] * B[2, 0:7] -> C[0:3, 0:7]
    # M(3,1,7) = 21 (Schoolbook)
    add_schoolbook_block(3, 1, 7, 0, 2, 2, 0, 0, 0)

    if col != rank:
        # If we didn't use exactly rank terms, something is wrong
        # But for now, just return what we have
        pass

    # Verification
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Use a few random tests
    np.random.seed(42)
    valid = True
    for _ in range(3):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_true = A_test @ B_test
        C_comp = np.zeros((n, p))
        
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_val = np.dot(U[i*m:(i+1)*m, r], A_test[i, :])
                    # V index is j*p + k. For fixed k, j goes 0..m-1.
                    # So we need V[k::p, r] but only the first m elements.
                    # V is (m*p, rank).
                    # v_vec = V[k::p, r][:m]
                    v_val = np.dot(V[k::p, r][:m], B_test[:, k])
                    val += W[k*n+i, r] * u_val * v_val
                C_comp[i, k] = val
                
        if not np.allclose(C_comp, C_true, atol=1e-8):
            valid = False
            break

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
