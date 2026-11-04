import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 120

    # We aim for rank 36: two independent 2x2x5 blocks.
    # Each 2x2x5 block uses Strassen (rank 7 for 2x2x2) + 4 trivial (for 2x2x1) = rank 11.
    # Total rank = 2 * 11 = 22... wait, that's not right.
    # Actually for 2x2x5: we need to handle 3 columns of B.
    # Strassen handles 2 columns (rank 7), then 1 more column needs rank 4 (schoolbook for 2x2x1).
    # So 2x2x5 has rank 7+4 = 11.
    # Two such blocks: 22. But that can't be right since schoolbook is 40.
    # Wait, the issue is that Strassen's 2x2x2 decomposition handles 2x2x2.
    # For 2x2x5, we have 5 columns in B. We can group them as 2+2+1.
    # 2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = rank 18.
    # Two blocks: 36. That matches.

    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        # Check all required conditions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Orthogonality checks
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j2 * p + k
                        idx_w = k * n + i
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-9:
                            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        for j2 in range(m):
                            if j2 == j:
                                continue
                            idx_u = i * m + j
                            idx_v = j2 * p + k
                            idx_w = k * n + i2
                            val = 0.0
                            for c in range(r):
                                val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                            if abs(val) > 1e-9:
                                return False
        return True

    def get_strassen_222(j_offset, k_offset):
        """Strassen 2x2x2 decomposition. j_offset: which pair of middle indices, k_offset: which pair of output columns."""
        U = np.zeros((n * m, 7))
        V = np.zeros((m * p, 7))
        W = np.zeros((p * n, 7))
        
        j0, j1 = j_offset, j_offset + 1
        k0, k1 = k_offset, k_offset + 1

        # M1 = (A00+A11)(B00+B11)
        U[0*m+j0, 0] = 1.0; U[1*m+j1, 0] = 1.0
        V[j0*p+k0, 0] = 1.0; V[j1*p+k1, 0] = 1.0
        W[k0*n+0, 0] = 1.0; W[k1*n+1, 0] = 1.0

        # M2 = (A10+A11)B00
        U[1*m+j0, 1] = 1.0; U[1*m+j1, 1] = 1.0
        V[j0*p+k0, 1] = 1.0
        W[k0*n+1, 1] = 1.0; W[k1*n+1, 1] = -1.0

        # M3 = A00(B01-B11)
        U[0*m+j0, 2] = 1.0
        V[j0*p+k1, 2] = 1.0; V[j1*p+k1, 2] = -1.0
        W[k1*n+0, 2] = 1.0; W[k1*n+1, 2] = 1.0

        # M4 = A11(B10-B00)
        U[1*m+j1, 3] = 1.0
        V[j1*p+k0, 3] = 1.0; V[j0*p+k0, 3] = -1.0
        W[k0*n+0, 3] = 1.0; W[k1*n+1, 3] = 1.0

        # M5 = (A00+A01)B11
        U[0*m+j0, 4] = 1.0; U[0*m+j1, 4] = 1.0
        V[j1*p+k1, 4] = 1.0
        W[k0*n+0, 4] = -1.0; W[k1*n+0, 4] = 1.0

        # M6 = (A10-A00)(B00+B01)
        U[1*m+j0, 5] = 1.0; U[0*m+j0, 5] = -1.0
        V[j0*p+k0, 5] = 1.0; V[j0*p+k1, 5] = 1.0
        W[k1*n+1, 5] = 1.0

        # M7 = (A01-A11)(B10+B11)
        U[0*m+j1, 6] = 1.0; U[1*m+j1, 6] = -1.0
        V[j1*p+k0, 6] = 1.0; V[j1*p+k1, 6] = 1.0
        W[k0*n+0, 6] = 1.0

        return U, V, W

    def get_trivial_221(j_offset, k_idx):
        """Schoolbook 2x2x1 decomposition (rank 4)."""
        U = np.zeros((n * m, 4))
        V = np.zeros((m * p, 4))
        W = np.zeros((p * n, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*m + (j_offset + j), col] = 1.0
                V[(j_offset + j)*p + k_idx, col] = 1.0
                W[k_idx*n + i, col] = 1.0
                col += 1
        return U, V, W

    # Build decomposition for each 2x4x5 block (splitting m into two groups of 2)
    all_U, all_V, all_W = [], [], []

    for j_base in [0, 2]:
        # 2x2x2 for k=0,1
        U1, V1, W1 = get_strassen_222(j_base, 0)
        all_U.append(U1); all_V.append(V1); all_W.append(W1)
        # 2x2x2 for k=2,3
        U2, V2, W2 = get_strassen_222(j_base, 2)
        all_U.append(U2); all_V.append(V2); all_W.append(W2)
        # 2x2x1 for k=4
        U3, V3, W3 = get_trivial_221(j_base, 4)
        all_U.append(U3); all_V.append(V3); all_W.append(W3)

    U_full = np.hstack(all_U)
    V_full = np.hstack(all_V)
    W_full = np.hstack(all_W)

    rank = U_full.shape[1]
    # rank should be 2*(7+7+4) = 36

    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback to schoolbook (rank 40)
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
