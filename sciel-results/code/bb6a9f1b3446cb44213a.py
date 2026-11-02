import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Block-Strassen: partition 6x6 into 2x2 blocks of 3x3
    # Strassen rank: 7 * (3*3*3) = 7 * 27 = 189
    nb = 2  # number of blocks per dimension
    bs = 3  # block size
    
    # Strassen steps: each step is (A_block_coeffs, B_block_coeffs)
    # Block indices: 0=00, 1=01, 2=10, 3=11
    # A block idx -> (row_block, col_block): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # B block idx -> (row_block, col_block): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    steps = [
        # (list of (a_block, a_coeff), list of (b_block, b_coeff))
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)]),                # M2
        ([(0, 1)], [(1, 1), (3, -1)]),               # M3
        ([(3, 1)], [(2, 1), (0, -1)]),               # M4
        ([(0, 1), (1, 1)], [(3, 1)]),                # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),       # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),       # M7
    ]
    
    # C combination: for each C block (0=C00, 1=C01, 2=C10, 3=C11),
    # which M_s (0-indexed) contribute with what sign
    # C00 = M1 - M4 + M6 - M7  -> {0:1, 3:-1, 5:1, 6:-1}
    # C01 = M3 + M5            -> {2:1, 4:1}
    # C10 = M2 + M4            -> {1:1, 3:1}
    # C11 = M1 - M2 + M3 + M7  -> {0:1, 1:-1, 2:1, 6:1}
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00
        {2: 1, 4: 1},                  # C01
        {1: 1, 3: 1},                  # C10
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11
    ]
    
    block_rank = bs * bs * bs  # 27
    total_rank = 7 * block_rank  # 189
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for (ai, a_sign) in a_coeffs:
            for (bi, b_sign) in b_coeffs:
                # A block ai: rows [ai//2 * bs, ai//2 * bs + bs), cols [ai%2 * bs, ai%2 * bs + bs)
                # B block bi: rows [bi//2 * bs, bi//2 * bs + bs), cols [bi%2 * bs, bi%2 * bs + bs)
                a_row_start = (ai // 2) * bs
                a_col_start = (ai % 2) * bs
                b_row_start = (bi // 2) * bs
                b_col_start = (bi % 2) * bs
                
                for i in range(bs):
                    for j in range(bs):
                        for k in range(bs):
                            gi = a_row_start + i  # global row in A / C
                            gj = a_col_start + j  # global col in A / row in B
                            gk = b_col_start + k  # global col in B / C
                    
                    # Determine which C block this (gi, gk) falls into
                    c_block_row = gi // bs
                    c_block_col = gk // bs
                    c_block_idx = c_block_row * 2 + c_block_col
                    
                    c_sign = c_combos[c_block_idx].get(s, 0)
                    if c_sign == 0:
                        continue
                    
                    col = s * block_rank + i * (bs * bs) + j * bs + k
                    
                    U[gi * m + gj, col] = a_sign
                    V[gj * p + gk, col] = b_sign
                    W[gk * n + gi, col] = c_sign
    
    # Verify the decomposition
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should be 1 for all i,k,j
    # But wait - the convention says W[k*n+i, r] where the output index is k*n+i
    # C[k,i] = sum_j A[k,j] * B[j,i]
    # So U[k*m+j, r] * V[j*p+i, r] * W[i*n+k, r] summed over r should be 1
    
    # Let's verify by checking the tensor directly
    # For the matrix multiplication tensor: T[a, b, c] = 1 where
    # a indexes A (row*col), b indexes B (row*col), c indexes C (col*row)
    # C[row, col] = sum_mid A[row, mid] * B[mid, col]
    # So T[A_idx, B_idx, C_idx] = 1 where A_idx = row*m+mid, B_idx = mid*p+col, C_idx = col*n+row
    
    # Check a sample of entries
    np.random.seed(123)
    sample_indices = []
    for _ in range(500):
        i = np.random.randint(0, n)
        k = np.random.randint(0, m)
        j = np.random.randint(0, p)
        sample_indices.append((i, k, j))
    
    valid = True
    for (i, k, j) in sample_indices:
        # U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        u_val = U[i * m + k, :]
        v_val = V[k * p + j, :]
        w_val = W[j * n + i, :]
        s_sum = np.sum(u_val * v_val * w_val)
        if abs(s_sum - 1.0) > 1e-10:
            valid = False
            break
    
    if not valid:
        # Fallback: schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=np.float64)
        V_school = np.zeros((m * p, rank_school), dtype=np.float64)
        W_school = np.zeros((p * n, rank_school), dtype=np.float64)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_school[i * m + k, col] = 1
                    V_school[k * p + j, col] = 1
                    W_school[j * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
