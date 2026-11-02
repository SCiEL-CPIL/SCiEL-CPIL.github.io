import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=float)
    V_school = np.zeros((mp, rank_school), dtype=float)
    W_school = np.zeros((pn, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_school[a_idx, col] = 1.0
                V_school[b_idx, col] = 1.0
                W_school[c_idx, col] = 1.0
                col += 1

    # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    rank_strassen = 6 * 7 + 4
    U = np.zeros((nm, rank_strassen), dtype=float)
    V = np.zeros((mp, rank_strassen), dtype=float)
    W = np.zeros((pn, rank_strassen), dtype=float)
    
    def A_idx(i, j):
        return i * m + j
    
    def B_idx(j, k):
        return j * p + k
    
    def C_idx(i, k):
        return k * n + i
    
    def add_strassen_block(k0, col_offset):
        # Strassen for 2x2 * 2x2
        # Products:
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # U coefficients (A matrix)
        # V coefficients (B matrix)
        # W coefficients (C matrix)
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1
        U[A_idx(0, 0), col_offset + 0] += 1
        U[A_idx(1, 1), col_offset + 0] += 1
        V[B_idx(0, k0), col_offset + 0] += 1
        V[B_idx(1, k0 + 1), col_offset + 0] += 1
        W[C_idx(0, k0), col_offset + 0] += 1
        W[C_idx(0, k0 + 1), col_offset + 0] += 1
        W[C_idx(1, k0), col_offset + 0] += 1
        W[C_idx(1, k0 + 1), col_offset + 0] += 1
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[A_idx(1, 0), col_offset + 1] += 1
        U[A_idx(1, 1), col_offset + 1] += 1
        V[B_idx(0, k0), col_offset + 1] += 1
        W[C_idx(1, k0), col_offset + 1] += 1
        W[C_idx(1, k0 + 1), col_offset + 1] -= 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[A_idx(0, 0), col_offset + 2] += 1
        V[B_idx(0, k0 + 1), col_offset + 2] += 1
        V[B_idx(1, k0 + 1), col_offset + 2] -= 1
        W[C_idx(0, k0 + 1), col_offset + 2] += 1
        W[C_idx(1, k0 + 1), col_offset + 2] += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[A_idx(1, 1), col_offset + 3] += 1
        V[B_idx(1, k0), col_offset + 3] += 1
        V[B_idx(0, k0), col_offset + 3] -= 1
        W[C_idx(0, k0), col_offset + 3] += 1
        W[C_idx(1, k0), col_offset + 3] += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[A_idx(0, 0), col_offset + 4] += 1
        U[A_idx(0, 1), col_offset + 4] += 1
        V[B_idx(1, k0 + 1), col_offset + 4] += 1
        W[C_idx(0, k0), col_offset + 4] -= 1
        W[C_idx(0, k0 + 1), col_offset + 4] += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[A_idx(1, 0), col_offset + 5] += 1
        U[A_idx(0, 0), col_offset + 5] -= 1
        V[B_idx(0, k0), col_offset + 5] += 1
        V[B_idx(0, k0 + 1), col_offset + 5] += 1
        W[C_idx(1, k0 + 1), col_offset + 5] += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[A_idx(0, 1), col_offset + 6] += 1
        U[A_idx(1, 1), col_offset + 6] -= 1
        V[B_idx(1, k0), col_offset + 6] += 1
        V[B_idx(1, k0 + 1), col_offset + 6] += 1
        W[C_idx(0, k0), col_offset + 6] += 1
    
    def add_single_block(k, col_offset):
        # Schoolbook for 2x2 * 2x1
        # C00 = A00*B0k + A01*B1k
        # C10 = A10*B0k + A11*B1k
        
        U[A_idx(0, 0), col_offset + 0] += 1
        V[B_idx(0, k), col_offset + 0] += 1
        W[C_idx(0, k), col_offset + 0] += 1
        
        U[A_idx(0, 1), col_offset + 1] += 1
        V[B_idx(1, k), col_offset + 1] += 1
        W[C_idx(0, k), col_offset + 1] += 1
        
        U[A_idx(1, 0), col_offset + 2] += 1
        V[B_idx(0, k), col_offset + 2] += 1
        W[C_idx(1, k), col_offset + 2] += 1
        
        U[A_idx(1, 1), col_offset + 3] += 1
        V[B_idx(1, k), col_offset + 3] += 1
        W[C_idx(1, k), col_offset + 3] += 1
    
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    add_single_block(12, col)
    col += 4
    
    assert col == rank_strassen, f"col={col}, R={rank_strassen}"
    
    # Verify
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_arr = np.array(target_indices)
    a_idx = target_arr[:, 0]
    b_idx = target_arr[:, 1]
    c_idx = target_arr[:, 2]
    U_t = U[a_idx, :]
    V_t = V[b_idx, :]
    W_t = W[c_idx, :]
    prod = U_t * V_t * W_t
    sums = np.sum(prod, axis=1)
    valid = np.all(np.abs(sums - 1.0) < 1e-9)
    
    if not valid:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
