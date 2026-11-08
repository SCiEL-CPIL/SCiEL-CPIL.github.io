import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen-based decomposition for 2x4x8
    # Split into 8 blocks of 2x2x2
    # 1 i-block, 2 j-blocks, 4 k-blocks
    # Each 2x2x2 block uses rank 7 (Strassen)
    # Total rank: 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen rank-7 decomposition for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen multiplications:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U entries (A part): 2x2 matrix, index i*2+j
    # V entries (B part): 2x2 matrix, index j*2+k
    # W entries (C part): 2x2 matrix, index k*2+i (trace convention)
    
    # For each Strassen multiplication r (0..6):
    # U[r]: 2x2 matrix (which entries of A are used)
    # V[r]: 2x2 matrix (which entries of B are used)
    # W[r]: 2x2 matrix (which entries of C are produced, with signs)
    
    strassen_terms = [
        # r=0: M1 = (A00+A11)*(B00+B11)
        # U: A00+A11 -> [(0,1),(3,1)]
        # V: B00+B11 -> [(0,1),(3,1)]
        # W: contributes to C00 (+1) and C11 (+1) -> W[k*2+i]: C00 is k=0,i=0 -> idx 0; C11 is k=1,i=1 -> idx 3
        # W: [(0,1),(3,1)]
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # r=1: M2 = (A10+A11)*B00
        # U: A10+A11 -> [(2,1),(3,1)]
        # V: B00 -> [(0,1)]
        # W: contributes to C10 (+1) -> k=0,i=1 -> idx 1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),
        # r=2: M3 = A00*(B10-B11)
        # U: A00 -> [(0,1)]
        # V: B10-B11 -> [(2,1),(3,-1)]
        # W: contributes to C01 (+1) and C11 (+1) -> C01: k=1,i=0 -> idx 2; C11: k=1,i=1 -> idx 3
        ([(0, 1)], [(2, 1), (3, -1)], [(2, 1), (3, 1)]),
        # r=3: M4 = A11*(B01-B00)
        # U: A11 -> [(3,1)]
        # V: B01-B00 -> [(1,1),(0,-1)]
        # W: contributes to C00 (+1) and C10 (+1) -> C00: k=0,i=0 -> idx 0; C10: k=0,i=1 -> idx 1
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)]),
        # r=4: M5 = (A00+A01)*B11
        # U: A00+A01 -> [(0,1),(1,1)]
        # V: B11 -> [(3,1)]
        # W: contributes to C00 (-1) and C01 (+1) -> C00: k=0,i=0 -> idx 0; C01: k=1,i=0 -> idx 2
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # r=5: M6 = (A10-A00)*(B00+B01)
        # U: A10-A00 -> [(2,1),(0,-1)]
        # V: B00+B01 -> [(0,1),(1,1)]
        # W: contributes to C11 (+1) -> k=1,i=1 -> idx 3
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # r=6: M7 = (A01-A11)*(B10+B11)
        # U: A01-A11 -> [(1,1),(3,-1)]
        # V: B10+B11 -> [(2,1),(3,1)]
        # W: contributes to C00 (+1) -> k=0,i=0 -> idx 0
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Verify Strassen decomposition for 2x2x2
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 2 + k
                        w_idx = k * 2 + i
                        for (idx, val) in strassen_terms[r][0]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in strassen_terms[r][1]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in strassen_terms[r][2]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not verify_strassen():
        # This should never happen, but fallback to trivial rank 64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Add Strassen blocks for 2x4x8
    # 1 i-block (i in {0,1}), 2 j-blocks (j in {0,1} and {2,3}), 4 k-blocks (k in {0,1}, {2,3}, {4,5}, {6,7})
    for j_block in range(2):  # j_block: 0 or 1
        for k_block in range(4):  # k_block: 0, 1, 2, 3
            base_rank = (j_block * 4 + k_block) * 7
            for r in range(7):
                cr = base_rank + r
                # U: local i in {0,1}, local j in {0,1} -> global i, global j = j_block*2 + local_j
                for (idx, val) in strassen_terms[r][0]:
                    li, lj = divmod(idx, 2)
                    global_i = li
                    global_j = j_block * 2 + lj
                    U[global_i * m + global_j][cr] += val
                # V: local j in {0,1}, local k in {0,1} -> global j = j_block*2 + local_j, global k = k_block*2 + local_k
                for (idx, val) in strassen_terms[r][1]:
                    lj, lk = divmod(idx, 2)
                    global_j = j_block * 2 + lj
                    global_k = k_block * 2 + lk
                    V[global_j * p + global_k][cr] += val
                # W: local k in {0,1}, local i in {0,1} -> global k = k_block*2 + local_k, global i = local_i
                for (idx, val) in strassen_terms[r][2]:
                    lk, li = divmod(idx, 2)
                    global_k = k_block * 2 + lk
                    global_i = li
                    W[global_k * n + global_i][cr] += val
    
    # Final verification with numpy
    import numpy as np
    A = np.random.randint(-3, 4, size=(n, m))
    B = np.random.randint(-3, 4, size=(m, p))
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to trivial rank 64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
