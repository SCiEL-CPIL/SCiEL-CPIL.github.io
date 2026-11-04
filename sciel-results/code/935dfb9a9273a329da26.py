import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130

    # Helper to verify the decomposition
    def verify(U, V, W):
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check all valid (i,j,k) positions
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
        # Check zero positions
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
        # Also check other zero positions where i or k mismatch
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

    def schoolbook():
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
        return U, V, W

    # Strassen 2x2x2 decomposition
    def get_strassen_222(j_offset, k_offset):
        U = np.zeros((n * m, 7))
        V = np.zeros((m * p, 7))
        W = np.zeros((p * n, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0*4+(0+j_offset), 0] = 1.0
        U[1*4+(1+j_offset), 0] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 0] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 0] = 1.0
        W[(0+k_offset)*2+0, 0] = 1.0
        W[(1+k_offset)*2+1, 0] = 1.0

        # M2 = (A10+A11)B00
        U[1*4+(0+j_offset), 1] = 1.0
        U[1*4+(1+j_offset), 1] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 1] = 1.0
        W[(0+k_offset)*2+1, 1] = 1.0
        W[(1+k_offset)*2+1, 1] = -1.0

        # M3 = A00(B01-B11)
        U[0*4+(0+j_offset), 2] = 1.0
        V[(0+j_offset)*5+(1+k_offset), 2] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 2] = -1.0
        W[(1+k_offset)*2+0, 2] = 1.0
        W[(1+k_offset)*2+1, 2] = 1.0

        # M4 = A11(B10-B00)
        U[1*4+(1+j_offset), 3] = 1.0
        V[(1+j_offset)*5+(0+k_offset), 3] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 3] = -1.0
        W[(0+k_offset)*2+0, 3] = 1.0
        W[(1+k_offset)*2+1, 3] = 1.0

        # M5 = (A00+A01)B11
        U[0*4+(0+j_offset), 4] = 1.0
        U[0*4+(1+j_offset), 4] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 4] = 1.0
        W[(0+k_offset)*2+0, 4] = -1.0
        W[(1+k_offset)*2+0, 4] = 1.0

        # M6 = (A10-A00)(B00+B01)
        U[1*4+(0+j_offset), 5] = 1.0
        U[0*4+(0+j_offset), 5] = -1.0
        V[(0+j_offset)*5+(0+k_offset), 5] = 1.0
        V[(0+j_offset)*5+(1+k_offset), 5] = 1.0
        W[(1+k_offset)*2+1, 5] = 1.0

        # M7 = (A01-A11)(B10+B11)
        U[0*4+(1+j_offset), 6] = 1.0
        U[1*4+(1+j_offset), 6] = -1.0
        V[(1+j_offset)*5+(0+k_offset), 6] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 6] = 1.0
        W[(0+k_offset)*2+0, 6] = 1.0

        return U, V, W

    # Trivial 2x2x1 decomposition (rank 4)
    def get_trivial_221(j_offset, k_offset):
        U = np.zeros((n * m, 4))
        V = np.zeros((m * p, 4))
        W = np.zeros((p * n, 4))
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*4 + (j_offset + j), col] = 1.0
                V[(j_offset + j)*5 + (k_offset), col] = 1.0
                W[(k_offset)*2 + i, col] = 1.0
                col += 1
        return U, V, W

    # Build full decomposition
    # Block 1: j in {0, 1}
    U1_s1, V1_s1, W1_s1 = get_strassen_222(0, 0) # k in {0, 1}
    U1_s2, V1_s2, W1_s2 = get_strassen_222(0, 2) # k in {2, 3}
    U1_t1, V1_t1, W1_t1 = get_trivial_221(0, 4)  # k in {4}
    
    U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
    V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
    W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

    # Block 2: j in {2, 3}
    U2_s1, V2_s1, W2_s1 = get_strassen_222(2, 0) # k in {0, 1}
    U2_s2, V2_s2, W2_s2 = get_strassen_222(2, 2) # k in {2, 3}
    U2_t1, V2_t1, W2_t1 = get_trivial_221(2, 4)  # k in {4}
    
    U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
    V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
    W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
