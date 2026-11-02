import numpy as np

DIMENSIONS = [2, 4, 7]


def _strassen_222():
    """Rank-7 decomposition of 2x2x2 matrix multiplication (Strassen).
    Returns U (4,7), V (4,7), W (4,7) with index conventions:
      U[i*2 + j], V[j*2 + k], W[k*2 + i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00 -> C10 and C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11) -> C01 and C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00) -> C00 and C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11 -> C00 and C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _school_221():
    """Rank-4 schoolbook decomposition of 2x2x1.
    U (4,4), V (2,4), W (2,4).
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    for c in range(4):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
    return U, V, W


def _school_222():
    """Rank-4 schoolbook decomposition of 2x2x2 (only 4 terms needed since p=2).
    Actually schoolbook for 2x2x2 needs 4 terms: C00=a00*b00+a01*b10, etc.
    But that's 8 terms. Wait, schoolbook 2x2x2 = 4 multiplications.
    """
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    # C00 = A00*B00 + A01*B10 -> need 2 terms
    # C01 = A00*B01 + A01*B11 -> 2 terms
    # C10 = A10*B00 + A11*B10 -> 2 terms
    # C11 = A10*B01 + A11*B11 -> 2 terms
    # Total 8 terms for schoolbook 2x2x2.
    # For 2x2x2, Strassen (7) < schoolbook (8), so we always use Strassen.
    # This function is not used but kept for completeness.
    return U, V, W


def _compose_block(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                   n_total, m_total, p_total,
                   i_off, j_off, k_off, col_start):
    """Place a sub-decomposition (n_sub x m_sub x p_sub) into the full tensor
    at offsets (i_off, j_off, k_off). Returns new column offset.
    
    Full index conventions:
      U[i*m_total + j], V[j*p_total + k], W[k*n_total + i]
    Sub index conventions:
      U_sub[i_sub*m_sub + j_sub], V_sub[j_sub*p_sub + k_sub], W_sub[k_sub*n_sub + i_sub]
    """
    r_sub = U_sub.shape[1]
    U = np.zeros((n_total * m_total, r_sub))
    V = np.zeros((m_total * p_total, r_sub))
    W = np.zeros((p_total * n_total, r_sub))

    for i_sub in range(n_sub):
        for j_sub in range(m_sub):
            U[(i_off + i_sub) * m_total + (j_off + j_sub), :] = U_sub[i_sub * m_sub + j_sub, :]

    for j_sub in range(m_sub):
        for k_sub in range(p_sub):
            V[(j_off + j_sub) * p_total + (k_off + k_sub), :] = V_sub[j_sub * p_sub + k_sub, :]

    for k_sub in range(p_sub):
        for i_sub in range(n_sub):
            W[(k_off + k_sub) * n_total + (i_off + i_sub), :] = W_sub[k_sub * n_sub + i_sub, :]

    return U, V, W, col_start + r_sub


def _build_2x2x7(n=2, m=2, p=7):
    """Build rank-25 decomposition of 2x2x7.
    Split p=7 into p1=4, p2=3.
    2x2x4 = two 2x2x2 Strassen blocks (rank 14).
    2x2x3 = one 2x2x2 Strassen + one 2x2x1 schoolbook (rank 11).
    Total rank = 25.
    """
    U_list = []
    V_list = []
    W_list = []

    # 2x2x4 part: split into two 2x2x2
    for k_start in [0, 2]:
        Us, Vs, Ws = _strassen_222()
        U_b, V_b, W_b, _ = _compose_block(Us, Vs, Ws, 2, 2, 2, n, m, p, 0, 0, k_start, 0)
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)

    # 2x2x3 part: one Strassen 2x2x2 + one schoolbook 2x2x1
    Us, Vs, Ws = _strassen_222()
    U_b, V_b, W_b, _ = _compose_block(Us, Vs, Ws, 2, 2, 2, n, m, p, 0, 0, 4, 0)
    U_list.append(U_b)
    V_list.append(V_b)
    W_list.append(W_b)

    Us, Vs, Ws = _school_221()
    U_b, V_b, W_b, _ = _compose_block(Us, Vs, Ws, 2, 2, 1, n, m, p, 0, 0, 6, 0)
    U_list.append(U_b)
    V_list.append(V_b)
    W_list.append(W_b)

    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W


def _build_2x4x7():
    """Build rank-50 decomposition of 2x4x7.
    Split m=4 into two m=2 blocks. Each block is a 2x2x7 decomposition.
    Total rank = 2 * 25 = 50.
    """
    n, m, p = 2, 4, 7

    U_list = []
    V_list = []
    W_list = []

    for j_off in [0, 2]:
        U_sub, V_sub, W_sub = _build_2x2x7()
        # Place into full tensor
        U_b = np.zeros((n * m, U_sub.shape[1]))
        V_b = np.zeros((m * p, V_sub.shape[1]))
        W_b = np.zeros((p * n, W_sub.shape[1]))

        for i in range(n):
            for j_sub in range(2):
                U_b[i * m + (j_off + j_sub), :] = U_sub[i * 2 + j_sub, :]

        for j_sub in range(2):
            for k in range(p):
                V_b[(j_off + j_sub) * p + k, :] = V_sub[j_sub * 2 + k, :]

        for k in range(p):
            for i in range(n):
                W_b[k * n + i, :] = W_sub[k * 2 + i, :]

        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)

    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W


def construct():
    U, V, W = _build_2x4x7()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
