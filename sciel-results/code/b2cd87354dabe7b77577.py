"""Tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses a recursive block decomposition with Strassen on 2x2x2 sub-blocks.
    Splits n=3 as 2+1, m=4 as 2+2, p=7 as 2+2+2+1.
    Target rank: 78 (better than schoolbook 84).
    """
    start_time = time.time()
    deadline = 110.0  # safety margin before 150s kill

    n, m, p = 3, 4, 7

    def schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        """Standard schoolbook decomposition for a sub-block."""
        U_rows = []
        V_rows = []
        W_rows = []
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[ii * m + jj] = 1.0
                    v[jj * p + kk] = 1.0
                    w[kk * n + ii] = 1.0
                    U_rows.append(u)
                    V_rows.append(v)
                    W_rows.append(w)
        return U_rows, V_rows, W_rows

    def verify_decomp(U, V, W, n, m, p):
        """Verify tensor identity: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all valid (i,j,k)."""
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_mat[i * m + j], V_mat[j * p + k] * W_mat[k * n + i]))
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def strassen_2x2(n_l, m_l, p_l, i_off, j_off, k_off):
        """
        Correct Strassen 2x2x2 decomposition.
        A indexed [A00, A01, A10, A11] (row-major)
        B indexed [B00, B01, B10, B11] (row-major)
        C indexed [C00, C01, C10, C11] (row-major)
        
        Standard Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B11)
        M4 = A11*(B01-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        """
        U_s = [
            [1, 0, 0, 1],    # M1: A00 + A11
            [0, 0, 1, 1],    # M2: A10 + A11
            [1, 0, 0, 0],    # M3: A00
            [0, 0, 0, 1],    # M4: A11
            [1, 1, 0, 0],    # M5: A00 + A01
            [-1, 0, 1, 0],   # M6: A10 - A00
            [0, 1, 0, -1],   # M7: A01 - A11
        ]
        V_s = [
            [1, 0, 0, 1],    # M1: B00 + B11
            [1, 0, 0, 0],    # M2: B00
            [0, 0, 1, -1],   # M3: B10 - B11
            [-1, 1, 0, 0],   # M4: B01 - B00
            [0, 0, 0, 1],    # M5: B11
            [1, 1, 0, 0],    # M6: B00 + B01
            [0, 0, 1, 1],    # M7: B10 + B11
        ]
        W_s = [
            [1, 0, 0, 1],    # M1: C00 + C11
            [0, 0, 1, 0],    # M2: C10
            [0, 1, 0, 0],    # M3: C01
            [1, 0, 1, 0],    # M4: C00 + C10
            [-1, 1, 0, 0],   # M5: -C00 + C01
            [0, 0, 0, 0],    # M6: not in C (valid for rank-7)
            [-1, 0, 0, 1],   # M7: -C00 + C11
        ]

        U_rows = []
        V_rows = []
        W_rows = []

        for r in range(7):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)

            # U vector: 4 entries for the 2x2 A block
            for dr in range(4):
                u_val = U_s[r][dr]
                if u_val != 0:
                    ri = dr // 2
                    cj = dr % 2
                    u[(i_off + ri) * m + (j_off + cj)] = u_val

            # V vector: 4 entries for the 2x2 B block
            for dc in range(4):
                v_val = V_s[r][dc]
                if v_val != 0:
                    ri = dc // 2
                    cj = dc % 2
                    v[(j_off + ri) * p + (k_off + cj)] = v_val

            # W vector: 4 entries for the 2x2 C block
            for dc in range(4):
                w_val = W_s[r][dc]
                if w_val != 0:
                    ri = dc // 2
                    cj = dc % 2
                    w[(k_off + ri) * n + (i_off + cj)] = w_val

            U_rows.append(u)
            V_rows.append(v)
            W_rows.append(w)

        return U_rows, V_rows, W_rows

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        """
        Recursively decompose the sub-tensor.
        Returns (U_rows, V_rows, W_rows) for this sub-block.
        """
        # Base cases
        if n_l == 0 or m_l == 0 or p_l == 0:
            return [], [], []

        if n_l == 1 and m_l == 1 and p_l == 1:
            return schoolbook(1, 1, 1, i_off, j_off, k_off)

        if n_l == 1:
            return schoolbook(1, m_l, p_l, i_off, j_off, k_off)

        if m_l == 1:
            return schoolbook(n_l, 1, p_l, i_off, j_off, k_off)

        if p_l == 1:
            return schoolbook(n_l, m_l, 1, i_off, j_off, k_off)

        # Strassen for 2x2x2
        if n_l == 2 and m_l == 2 and p_l == 2:
            return strassen_2x2(2, 2, 2, i_off, j_off, k_off)

        # Split strategy for larger blocks
        # For 3x4x7: split n=3 as 2+1
        if n_l == 3 and m_l == 4 and p_l == 7:
            U1, V1, W1 = gen_terms(2, 4, 7, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return U1 + U2, V1 + V2, W1 + W2

        # For 2x4x7: split m=4 as 2+2
        if n_l == 2 and m_l == 4 and p_l == 7:
            U1, V1, W1 = gen_terms(2, 2, 7, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return U1 + U2, V1 + V2, W1 + W2

        # For 2x2x7: split p=7 as 2+2+2+1
        if n_l == 2 and m_l == 2 and p_l == 7:
            U1, V1, W1 = gen_terms(2, 2, 2, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            U3, V3, W3 = gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            U4, V4, W4 = gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return (U1 + U2 + U3 + U4,
                    V1 + V2 + V3 + V4,
                    W1 + W2 + W3 + W4)

        # For 2x2x3: split p=3 as 2+1
        if n_l == 2 and m_l == 2 and p_l == 3:
            U1, V1, W1 = gen_terms(2, 2, 2, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(2, 2, 1, i_off, j_off, k_off + 2)
            return U1 + U2, V1 + V2, W1 + W2

        # For 2x3x2: split m=3 as 2+1
        if n_l == 2 and m_l == 3 and p_l == 2:
            U1, V1, W1 = gen_terms(2, 2, 2, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(2, 1, 2, i_off, j_off + 2, k_off)
            return U1 + U2, V1 + V2, W1 + W2

        # For 3x2x2: split n=3 as 2+1
        if n_l == 3 and m_l == 2 and p_l == 2:
            U1, V1, W1 = gen_terms(2, 2, 2, i_off, j_off, k_off)
            U2, V2, W2 = gen_terms(1, 2, 2, i_off + 2, j_off, k_off)
            return U1 + U2, V1 + V2, W1 + W2

        # Fallback: schoolbook
        return schoolbook(n_l, m_l, p_l, i_off, j_off, k_off)

    # Try the recursive construction
    if time.time() - start_time < deadline - 5:
        U_rows, V_rows, W_rows = gen_terms(n, m, p, 0, 0, 0)
        rank = len(U_rows)

        if rank < n * m * p:
            # Verify
            if verify_decomp(U_rows, V_rows, W_rows, n, m, p):
                return {"U": U_rows, "V": V_rows, "W": W_rows}
            else:
                # Verification failed, fall back to schoolbook
                U_s, V_s, W_s = schoolbook(n, m, p, 0, 0, 0)
                return {"U": U_s, "V": V_s, "W": W_s}

    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook(n, m, p, 0, 0, 0)
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
