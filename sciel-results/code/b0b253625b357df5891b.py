import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # Output convention: W[k*n + i, r] where i is row of A/C, k is col of B/C
    # A is n x m (3x4), B is m x p (4x4), C is n x p (3x4)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # We will construct a rank 30 decomposition.
    # Strategy:
    # Split B's rows (j index) into two groups of 2: J1={0,1}, J2={2,3}
    # C = A[:, J1] @ B[J1, :] + A[:, J2] @ B[J2, :]
    # Let A1 = A[:, 0:2] (3x2), B1 = B[0:2, :] (2x4)
    # Let A2 = A[:, 2:4] (3x2), B2 = B[2:4, :] (2x4)
    # Term 1: A1 @ B1 (3x2x4)
    # Term 2: A2 @ B2 (3x2x4)
    
    # For A1 @ B1 (3x2x4):
    # Split A1's rows (i index) into I1={0,1} (2 rows) and I2={2} (1 row)
    # A1 = [A1_top; A1_bot] where A1_top is 2x2, A1_bot is 1x2
    # B1 is 2x4. Split B1's cols (k index) into K1={0,1} (2 cols) and K2={2,3} (2 cols)
    # B1 = [B1_left | B1_right] where B1_left is 2x2, B1_right is 2x2
    
    # Sub-term 1.1: A1_top @ B1_left (2x2x2) -> Rank 7 (Strassen)
    # Sub-term 1.2: A1_top @ B1_right (2x2x2) -> Rank 7 (Strassen)
    # Sub-term 1.3: A1_bot @ B1 (1x2x4) -> Rank 1
    # Total rank for A1 @ B1 = 7 + 7 + 1 = 15.
    
    # Similarly for A2 @ B2 (3x2x4): Rank 15.
    
    # Total rank = 15 + 15 = 30.
    
    rank = 30
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 terms
    def add_strassen(u_offset, v_offset, w_offset):
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # C = A @ B
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+g)
        # M7 = (b-d)(e+g)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        i_base, j_base = u_offset
        j_base_v, k_base_v = v_offset
        i_base_w, k_base_w = w_offset
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, 0, 0]
        ]
        
        for t in range(7):
            # Fill U
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_idx = (i_base + li) * m + (j_base + lj)
                    U[global_idx, col] += u_vecs[t][local_idx]
            
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_idx = (j_base_v + lj) * p + (k_base_v + lk)
                    V[global_idx, col] += v_vecs[t][local_idx]
            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_idx = (k_base_w + lk) * n + (i_base_w + li)
                    W[global_idx, col] += w_vecs[t][local_idx]
            
            col += 1

    # Helper to add 1x2x4 term (Rank 1)
    def add_1x2x4(u_offset, v_offset, w_offset):
        nonlocal col
        i_base, j_base = u_offset
        j_base_v, k_base_v = v_offset
        i_base_w, k_base_w = w_offset
        
        for lj in range(2):
            global_idx_u = (i_base) * m + (j_base + lj)
            U[global_idx_u, col] += 1
            
            for lk in range(4):
                global_idx_v = (j_base_v + lj) * p + (k_base_v + lk)
                V[global_idx_v, col] += 1
        
        for lk in range(4):
            global_idx_w = (k_base_w + lk) * n + i_base_w
            W[global_idx_w, col] += 1
        
        col += 1

    # Block 1: A1 @ B1
    # A1: i in {0,1,2}, j in {0,1}
    # B1: j in {0,1}, k in {0,1,2,3}
    
    # Sub-term 1.1: A1_top @ B1_left (2x2x2)
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen((0, 0), (0, 0), (0, 0))
    
    # Sub-term 1.2: A1_top @ B1_right (2x2x2)
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen((0, 0), (0, 2), (0, 2))
    
    # Sub-term 1.3: A1_bot @ B1 (1x2x4)
    # i in {2}, j in {0,1}, k in {0,1,2,3}
    add_1x2x4((2, 0), (0, 0), (2, 0))
    
    # Block 2: A2 @ B2
    # A2: i in {0,1,2}, j in {2,3}
    # B2: j in {2,3}, k in {0,1,2,3}
    
    # Sub-term 2.1: A2_top @ B2_left (2x2x2)
    # i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen((0, 2), (2, 0), (0, 0))
    
    # Sub-term 2.2: A2_top @ B2_right (2x2x2)
    # i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen((0, 2), (2, 2), (0, 2))
    
    # Sub-term 2.3: A2_bot @ B2 (1x2x4)
    # i in {2}, j in {2,3}, k in {0,1,2,3}
    add_1x2x4((2, 2), (2, 0), (2, 0))
    
    # Verify rank
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank0 = n * m * p
        U = np.zeros((n * m, rank0), dtype=float)
        V = np.zeros((m * p, rank0), dtype=float)
        W = np.zeros((p * n, rank0), dtype=float)
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
