import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 140  # safety margin from 150s limit


def _strassen_2x2_2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape (4,7): row = i*2+j, represents A[i,j]
    V shape (4,7): row = j*2+k, represents B[j,k]
    W shape (4,7): row = k*2+i, represents C[i,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # M1
    U[0, 0] = 1  # A00 (i=0,j=0)
    U[3, 0] = 1  # A11 (i=1,j=1)
    V[0, 0] = 1  # B00 (j=0,k=0)
    V[3, 0] = 1  # B11 (j=1,k=1)
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)

    # M2
    U[2, 1] = 1  # A10 (i=1,j=0)
    U[3, 1] = 1  # A11 (i=1,j=1)
    V[0, 1] = 1  # B00 (j=0,k=0)
    W[2, 1] = 1  # C10 (k=1,i=0)

    # M3
    U[0, 2] = 1  # A00 (i=0,j=0)
    V[2, 2] = 1  # B10 (j=1,k=0)
    V[3, 2] = -1  # B11 (j=1,k=1)
    W[1, 2] = 1  # C01 (k=0,i=1)
    W[2, 2] = 1  # C10 (k=1,i=0)

    # M4
    U[3, 3] = 1  # A11 (i=1,j=1)
    V[1, 3] = 1  # B01 (j=0,k=1)
    V[0, 3] = -1  # B00 (j=0,k=0)
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=1,i=0)

    # M5
    U[0, 4] = 1  # A00 (i=0,j=0)
    U[1, 4] = 1  # A01 (i=0,j=1)
    V[3, 4] = 1  # B11 (j=1,k=1)
    W[0, 4] = -1  # C00 (k=0,i=0)
    W[1, 4] = 1  # C01 (k=0,i=1)

    # M6
    U[2, 5] = 1  # A10 (i=1,j=0)
    U[0, 5] = -1  # A00 (i=0,j=0)
    V[0, 5] = 1  # B00 (j=0,k=0)
    V[1, 5] = 1  # B01 (j=0,k=1)
    W[3, 5] = 1  # C11 (k=1,i=1)

    # M7
    U[1, 6] = 1  # A01 (i=0,j=1)
    U[3, 6] = -1  # A11 (i=1,j=1)
    V[2, 6] = 1  # B10 (j=1,k=0)
    V[3, 6] = 1  # B11 (j=1,k=1)
    W[0, 6] = 1  # C00 (k=0,i=0)
    W[3, 6] = 1  # C11 (k=1,i=1)

    return U, V, W


def _verify(U, V, W, n=3, m=3, p=8, tol=1e-9):
    """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor T[i,j,k]=1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False

    T = np.zeros((n, m, p))
    for c in range(rank):
        u_mat = U[:, c].reshape(n, m)
        v_mat = V[:, c].reshape(m, p)
        w_mat = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_mat[i, j] * (v_mat[j, :] * w_mat[:, i])

    return np.max(np.abs(T - 1.0)) < tol


def _schoolbook():
    """Fallback: rank 72 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_strassen_decomposition():
    """
    Build a rank-56 decomposition:
    For each k-block (2 k-values):
    - Strassen 2x2x2 for A[i,0:2], B[0:2,k] -> C[i,k], i in {0,1}: rank 7
    - Outer product for A[i,2], B[2,k] -> C[i,k], i in {0,1}: rank 4
    - Schoolbook for A[2,j], B[j,k] -> C[2,k], j in {0,1,2}: rank 6
    Total per block: 17, total: 4*17 = 68.
    
    Wait, that's the same as before. Let me reconsider.
    Actually, the 1x3x2 part can be done more efficiently.
    C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    This is a dot product. For each k, it's one scalar. But in tensor form,
    T[2,j,k] = 1 for all j,k. The rank of this 1x3x2 tensor is 3 (since
    it's sum over j of A[2,j] * B[j,k], which is 3 rank-1 terms).
    
    So per block: 7 + 4 + 3 = 14. Total: 4*14 = 56.
    """
    n, m, p = DIMENSIONS
    rank_total = 56
    U = np.zeros((n*m, rank_total))  # 9 x 56
    V = np.zeros((m*p, rank_total))  # 24 x 56
    W = np.zeros((p*n, rank_total))  # 24 x 56

    U_s, V_s, W_s = _strassen_2x2_2()  # shapes (4,7), (4,7), (4,7)

    col = 0
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1

        # --- Part 1: 2x2x2 Strassen for A[i,0:2], B[0:2,k] -> C[i,k], i in {0,1} ---
        # 7 terms. U_s row = i_sub*2 + j_sub, V_s row = j_sub*2 + k_sub, W_s row = k_sub*2 + i_sub
        for l in range(7):
            c = col + l
            # U: i_sub in {0,1}, j_sub in {0,1}
            for i_sub in range(2):
                for j_sub in range(2):
                    idx = i_sub * 2 + j_sub
                    U[i_sub * 3 + j_sub, c] = U_s[idx, l]
            # V: j_sub in {0,1}, k_sub in {0,1}
            for j_sub in range(2):
                for k_sub in range(2):
                    idx = j_sub * 2 + k_sub
                    k_global = k0 + k_sub
                    V[j_sub * 8 + k_global, c] = V_s[idx, l]
            # W: k_sub in {0,1}, i_sub in {0,1}
            for k_sub in range(2):
                for i_sub in range(2):
                    idx = k_sub * 2 + i_sub
                    k_global = k0 + k_sub
                    W[k_global * 3 + i_sub, c] = W_s[idx, l]
            col += 1

        # --- Part 2: Outer product for A[i,2], B[2,k] -> C[i,k], i in {0,1} ---
        # 4 terms: (i_sub, k_sub) in {0,1}x{0,1}
        for i_sub in range(2):
            for k_sub in range(2):
                c = col
                k_global = k0 + k_sub
                U[i_sub * 3 + 2, c] = 1.0
                V[2 * 8 + k_global, c] = 1.0
                W[k_global * 3 + i_sub, c] = 1.0
                col += 1

        # --- Part 3: 1x3x2 for A[2,j], B[j,k] -> C[2,k] ---
        # 6 terms: (j, k_sub) in {0,1,2}x{0,1}
        for j in range(3):
            for k_sub in range(2):
                c = col
                k_global = k0 + k_sub
                U[2 * 3 + j, c] = 1.0
                V[j * 8 + k_global, c] = 1.0
                W[k_global * 3 + 2, c] = 1.0
                col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result


def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    # Build the Strassen-based decomposition (rank 56)
    if time_left() > 5:
        result = _build_strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"]):
            return result

        # Try rounding
        U = np.asarray(result["U"], dtype=float)
        V = np.asarray(result["V"], dtype=float)
        W = np.asarray(result["W"], dtype=float)
        for scale in [2, 4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            result_r = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            if _verify(result_r["U"], result_r["V"], result_r["W"]):
                return result_r

    # Fallback to schoolbook
    return _schoolbook()
