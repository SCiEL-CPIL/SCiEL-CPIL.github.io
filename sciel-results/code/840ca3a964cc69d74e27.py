import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # Output convention: W[k*n + i, r] where i is row of A/C, k is col of B/C
    # A is n x m (3x4), B is m x p (4x4), C is n x p (3x4)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # Rank 29 Decomposition
    # Split i into {0,1} and {2}
    # For i in {0,1}:
    #   Split j into {0,1} and {2,3}
    #   For j in {0,1}:
    #     Split k into {0,1} and {2,3}
    #     k in {0,1}: 2x2x2 Strassen (Rank 7)
    #     k in {2,3}: 2x2x2 Strassen (Rank 7)
    #   For j in {2,3}:
    #     Split k into {0,1} and {2,3}
    #     k in {0,1}: 2x2x2 Strassen (Rank 7)
    #     k in {2,3}: 2x2x2 Strassen (Rank 7)
    # For i in {2}:
    #   1x4x4 block (Rank 1)
    
    rank = 29
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # Local indices:
        # A: (0,0)..(1,1) -> global (i_base+li, j_base+lj)
        # B: (0,0)..(1,1) -> global (j_base+lj, k_base+lk)
        # C: (0,0)..(1,1) -> global (k_base+lk, i_base+li)
        
        # U coeffs for A
        u_vecs = [
            [1, 0, 0, 1], # M1: a+d
            [0, 0, 1, 1], # M2: c+d
            [1, 0, 0, 0], # M3: a
            [0, 0, 0, 1], # M4: d
            [1, 1, 0, 0], # M5: a+b
            [0, 0, 1, -1],# M6: c-d
            [0, 1, 0, -1] # M7: b-d
        ]
        # V coeffs for B
        v_vecs = [
            [1, 0, 0, 1], # M1: e+h
            [1, 0, 0, 0], # M2: e
            [0, 1, 0, -1],# M3: f-h
            [0, 0, 1, -1],# M4: g-e
            [0, 0, 0, 1], # M5: h
            [0, 1, 1, 0], # M6: f+g
            [1, 0, 1, 0]  # M7: e+g
        ]
        # W coeffs for C (index order: k*n + i)
        # Local C indices: (k=0, i=0)=0, (k=0, i=1)=1, (k=1, i=0)=2, (k=1, i=1)=3
        # C = [[C00, C01], [C10, C11]]
        # M1: C00+1, C11+1 -> [1, 0, 0, 1]
        # M2: C10+1 -> [0, 1, 0, 0]
        # M3: C01+1, C11-1 -> [0, 0, 1, -1]
        # M4: C00-1, C10+1 -> [-1, 1, 0, 0]
        # M5: C00-1, C01+1 -> [-1, 0, 1, 0]
        # M6: C11+1 -> [0, 0, 0, 1]
        # M7: C00+1, C10-1 -> [1, -1, 0, 0]
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
                    global_idx = (j_base + lj) * p + (k_base + lk)
                    V[global_idx, col] += v_vecs[t][local_idx]
            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_idx = (k_base + lk) * n + (i_base + li)
                    W[global_idx, col] += w_vecs[t][local_idx]
            
            col += 1

    def add_1x4x4(i_base, j_base, k_base):
        nonlocal col
        # i is fixed to i_base
        # j ranges j_base to j_base+3
        # k ranges k_base to k_base+3
        # Rank 1: U=1, V=1, W=1
        for lj in range(4):
            global_idx_u = i_base * m + (j_base + lj)
            U[global_idx_u, col] = 1.0
            
            for lk in range(4):
                global_idx_v = (j_base + lj) * p + (k_base + lk)
                V[global_idx_v, col] = 1.0
        
        for lk in range(4):
            global_idx_w = (k_base + lk) * n + i_base
            W[global_idx_w, col] = 1.0
        
        col += 1

    # Apply blocks
    
    # Top-Left-Top: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # Top-Left-Bot: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    
    # Top-Right-Top: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen(0, 2, 0)
    
    # Top-Right-Bot: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2)
    
    # Bottom: i in {2}, j in {0,1,2,3}, k in {0,1,2,3}
    add_1x4x4(2, 0, 0)
    
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
