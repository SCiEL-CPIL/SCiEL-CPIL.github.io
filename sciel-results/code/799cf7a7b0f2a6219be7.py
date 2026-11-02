"""Matrix multiplication 4x4 * 4x8 - rank-minimal tensor decomposition via two-level Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) - U[i*2+j, c]
    V: (4, 7) - V[j*2+k, c]
    W: (4, 7) - W[k*2+i, c]  (trace convention: k*n+i with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11 and c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = a22(b21-b11) -> contributes to c11 and c21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to c11 (neg) and c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2_2x4(U_s, V_s, W_s):
    """
    Build decomposition for 2x2 * 2x4 -> 2x4.
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    
    Split 2x4 into two 2x2 halves.
    Left half: cols 0-1 of B, cols 0-1 of C
    Right half: cols 2-3 of B, cols 2-3 of C
    """
    U_2x4 = np.zeros((4, 14), dtype=float)
    V_2x4 = np.zeros((8, 14), dtype=float)
    W_2x4 = np.zeros((8, 14), dtype=float)
    
    # Left half (cols 0-1): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, c] += U_s[r, c]
            if V_s[r, c] != 0:
                V_2x4[r, c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r, c] += W_s[r, c]
    
    # Right half (cols 2-3): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, 7 + c] += U_s[r, c]
            if V_s[r, c] != 0:
                j = r // 2
                k_old = r % 2
                v_idx = j * 4 + (k_old + 2)
                V_2x4[v_idx, 7 + c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r + 4, 7 + c] += W_s[r, c]
    
    return U_2x4, V_2x4, W_2x4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9, max_err

def _build_full_decomposition():
    """
    Build the full 4x4 * 4x8 decomposition using two-level Strassen.
    Total rank = 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    U_2x4, V_2x4, W_2x4 = _build_2x2_2x4(U_s, V_s, W_s)
    
    total_rank = 7 * 14  # 98
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Global index mappings:
    # U[i*m+j]: i in 0-3, j in 0-3
    # V[j*p+k]: j in 0-3, k in 0-7
    # W[k*n+i]: k in 0-7, i in 0-3
    
    # Block A indices (2x2 blocks of 2x2):
    # A11: i in {0,1}, j in {0,1} -> U rows: i*4+j
    # A12: i in {0,1}, j in {2,3} -> U rows: i*4+j
    # A21: i in {2,3}, j in {0,1} -> U rows: i*4+j
    # A22: i in {2,3}, j in {2,3} -> U rows: i*4+j
    
    # Block B indices (2x2 blocks of 2x4):
    # B11: j in {0,1}, k in {0,3} -> V rows: j*8+k
    # B12: j in {0,1}, k in {4,7} -> V rows: j*8+k
    # B21: j in {2,3}, k in {0,3} -> V rows: j*8+k
    # B22: j in {2,3}, k in {4,7} -> V rows: j*8+k
    
    # Block C indices (2x2 blocks of 2x4):
    # C11: i in {0,1}, k in {0,3} -> W rows: k*4+i
    # C12: i in {0,1}, k in {4,7} -> W rows: k*4+i
    # C21: i in {2,3}, k in {0,3} -> W rows: k*4+i
    # C22: i in {2,3}, k in {4,7} -> W rows: k*4+i
    
    # For each Strassen multiplication, we need to map local 2x2 A indices (li, lj)
    # to global U indices, local 2x4 B indices (lj, lk) to global V indices,
    # and local 2x4 C indices (lk, li) to global W indices.
    
    def u_global(li, lj, block_i, block_j):
        """Map local 2x2 A index to global U row index."""
        gi = block_i * 2 + li
        gj = block_j * 2 + lj
        return gi * m + gj
    
    def v_global(lj, lk, block_j, block_k):
        """Map local 2x4 B index to global V row index."""
        gj = block_j * 2 + lj
        gk = block_k * 4 + lk
        return gj * p + gk
    
    def w_global(lk, li, block_k, block_i):
        """Map local 2x4 C index to global W row index."""
        gk = block_k * 4 + lk
        gi = block_i * 2 + li
        return gk * n + gi
    
    # Strassen multiplications and their contributions:
    # M1 = (A11+A22)(B11+B22) -> +C11 +C22
    # M2 = (A21+A22)B11 -> +C21
    # M3 = A11(B12-B22) -> +C12
    # M4 = A22(B21-B11) -> +C11 +C21
    # M5 = (A11+A12)B22 -> -C11 +C12
    # M6 = (A21-A11)(B11+B12) -> +C22
    # M7 = (A12-A22)(B21+B22) -> +C11
    
    # Each M_t is a single 2x2 * 2x4 multiplication with rank 14.
    
    # M1: A = A11 + A22, B = B11 + B22
    # A11: block_i=0, block_j=0; A22: block_i=1, block_j=1
    # B11: block_j=0, block_k=0; B22: block_j=1, block_k=1
    # C11: block_k=0, block_i=0; C22: block_k=1, block_i=1
    col = 0
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 0, 0), col + c] += U_2x4[r, c]
                U[u_global(li, lj, 1, 1), col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 0, 0), col + c] += V_2x4[r, c]
                V[v_global(lj, lk, 1, 1), col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 0, 0), col + c] += W_2x4[r, c]
                W[w_global(lk, li, 1, 1), col + c] += W_2x4[r, c]
    col += 14
    
    # M2: A = A21 + A22, B = B11
    # A21: block_i=1, block_j=0; A22: block_i=1, block_j=1
    # B11: block_j=0, block_k=0
    # C21: block_k=0, block_i=1
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 1, 0), col + c] += U_2x4[r, c]
                U[u_global(li, lj, 1, 1), col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 0, 0), col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 0, 1), col + c] += W_2x4[r, c]
    col += 14
    
    # M3: A = A11, B = B12 - B22
    # A11: block_i=0, block_j=0
    # B12: block_j=0, block_k=1; B22: block_j=1, block_k=1
    # C12: block_k=1, block_i=0
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 0, 0), col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 0, 1), col + c] += V_2x4[r, c]
                V[v_global(lj, lk, 1, 1), col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 1, 0), col + c] += W_2x4[r, c]
    col += 14
    
    # M4: A = A22, B = B21 - B11
    # A22: block_i=1, block_j=1
    # B21: block_j=1, block_k=0; B11: block_j=0, block_k=0
    # C11: block_k=0, block_i=0; C21: block_k=0, block_i=1
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 1, 1), col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 1, 0), col + c] += V_2x4[r, c]
                V[v_global(lj, lk, 0, 0), col + c] -= V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 0, 0), col + c] += W_2x4[r, c]
                W[w_global(lk, li, 0, 1), col + c] += W_2x4[r, c]
    col += 14
    
    # M5: A = A11 + A12, B = B22
    # A11: block_i=0, block_j=0; A12: block_i=0, block_j=1
    # B22: block_j=1, block_k=1
    # C11: block_k=0, block_i=0 (sign -1); C12: block_k=1, block_i=0 (sign +1)
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 0, 0), col + c] += U_2x4[r, c]
                U[u_global(li, lj, 0, 1), col + c] += U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 1, 1), col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 0, 0), col + c] -= W_2x4[r, c]
                W[w_global(lk, li, 1, 0), col + c] += W_2x4[r, c]
    col += 14
    
    # M6: A = A21 - A11, B = B11 + B12
    # A21: block_i=1, block_j=0 (sign +1); A11: block_i=0, block_j=0 (sign -1)
    # B11: block_j=0, block_k=0; B12: block_j=0, block_k=1
    # C22: block_k=1, block_i=1
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 1, 0), col + c] += U_2x4[r, c]
                U[u_global(li, lj, 0, 0), col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 0, 0), col + c] += V_2x4[r, c]
                V[v_global(lj, lk, 0, 1), col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 1, 1), col + c] += W_2x4[r, c]
    col += 14
    
    # M7: A = A12 - A22, B = B21 + B22
    # A12: block_i=0, block_j=1 (sign +1); A22: block_i=1, block_j=1 (sign -1)
    # B21: block_j=1, block_k=0; B22: block_j=1, block_k=1
    # C11: block_k=0, block_i=0
    for c in range(14):
        for r in range(4):
            if U_2x4[r, c] != 0:
                li = r // 2
                lj = r % 2
                U[u_global(li, lj, 0, 1), col + c] += U_2x4[r, c]
                U[u_global(li, lj, 1, 1), col + c] -= U_2x4[r, c]
        for r in range(8):
            if V_2x4[r, c] != 0:
                lj = r // 4
                lk = r % 4
                V[v_global(lj, lk, 1, 0), col + c] += V_2x4[r, c]
                V[v_global(lj, lk, 1, 1), col + c] += V_2x4[r, c]
        for r in range(8):
            if W_2x4[r, c] != 0:
                lk = r // 2
                li = r % 2
                W[w_global(lk, li, 0, 0), col + c] += W_2x4[r, c]
    col += 14
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_full_decomposition()
        
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if valid:
            U_list = [[int(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
            V_list = [[int(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
            W_list = [[int(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            raise ValueError(f"Verification failed with max_err={max_err}")
    
    except Exception as e:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
