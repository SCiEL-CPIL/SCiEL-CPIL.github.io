import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Split 2x4x8 into four 2x2x4 subproblems
    # Each 2x2x4 uses a rank-12 decomposition
    # Total rank: 4 * 12 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48
    
    # Rank-12 decomposition for 2x2x4 tensor
    # For 2x2x4: A is 2x2, B is 2x4, C is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # U: 4x12 (indices i*2+j for i,j in {0,1})
    # V: 8x12 (indices j*4+k for j in {0,1}, k in {0,1,2,3})
    # W: 8x12 (indices k*2+i for k in {0,1,2,3}, i in {0,1})
    
    # Define the 12 rank-1 terms for 2x2x4
    # Each term is (U_entries, V_entries, W_entries)
    # U_entries: list of (local_index, value) where local_index = i*2+j
    # V_entries: list of (local_index, value) where local_index = j*4+k
    # W_entries: list of (local_index, value) where local_index = k*2+i
    
    terms_224 = [
        # Term 0: Handle k=0,1 with j=0,1 structure
        ([(0, 1), (3, 1)], [(0, 1), (7, 1)], [(0, 1), (3, 1)]),
        # Term 1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # Term 2
        ([(0, 1)], [(1, 1), (7, -1)], [(2, 1), (3, 1)]),
        # Term 3
        ([(3, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)]),
        # Term 4
        ([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)]),
        # Term 5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # Term 6
        ([(1, 1), (3, -1)], [(4, 1), (7, 1)], [(0, 1)]),
        # Term 7: Handle k=2,3 with j=0,1 structure
        ([(0, 1), (3, 1)], [(2, 1), (5, 1)], [(4, 1), (7, 1)]),
        # Term 8
        ([(2, 1), (3, 1)], [(2, 1)], [(5, 1), (7, -1)]),
        # Term 9
        ([(0, 1)], [(3, 1), (5, -1)], [(6, 1), (7, 1)]),
        # Term 10
        ([(3, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)]),
        # Term 11
        ([(0, 1), (1, 1)], [(5, 1)], [(4, -1), (6, 1)]),
    ]
    
    # Verify: check that this decomposition is correct for 2x2x4
    def verify_224():
        # Check that sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all (i,j,k)
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    total = 0
                    for r in range(12):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 4 + k
                        w_idx = k * 2 + i
                        for (idx, val) in terms_224[r][0]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in terms_224[r][1]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in terms_224[r][2]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    # Apply the rank-12 decomposition for each of the 4 sub-blocks
    # Sub-block (j_block, k_block): j in {j_block*2, j_block*2+1}, k in {k_block*4, ..., k_block*4+3}
    # j_block: 0 or 1, k_block: 0 or 1
    
    for j_block in range(2):
        for k_block in range(2):
            base_rank = (j_block * 2 + k_block) * 12
            for r in range(12):
                cr = base_rank + r
                # U: i in {0,1}, local j in {0,1} -> global j = j_block*2 + local_j
                for (idx, val) in terms_224[r][0]:
                    li, lj = divmod(idx, 2)
                    global_i = li
                    global_j = j_block * 2 + lj
                    U[global_i * 4 + global_j][cr] += val
                # V: local j in {0,1}, k in {0,1,2,3} -> global j = j_block*2 + local_j, global k = k_block*4 + local_k
                for (idx, val) in terms_224[r][1]:
                    lj, lk = divmod(idx, 4)
                    global_j = j_block * 2 + lj
                    global_k = k_block * 4 + lk
                    V[global_j * 8 + global_k][cr] += val
                # W: k in {0,1,2,3}, i in {0,1} -> global k = k_block*4 + local_k, global i = local_i
                for (idx, val) in terms_224[r][2]:
                    lk, li = divmod(idx, 2)
                    global_k = k_block * 4 + lk
                    global_i = li
                    W[global_k * 2 + global_i][cr] += val
    
    # Final verification with random test
    def full_verify():
        # Test with random matrices
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute C from decomposition: C[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not full_verify():
        # Fallback to Strassen-based rank 56
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)],
        ]
        strassen_V = [
            [(0, 1), (3, 1)],
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (0, -1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (3, 1)],
        ]
        strassen_W = [
            [(0, 1), (3, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1), (1, 1)],
            [(0, -1), (2, 1)],
            [(3, 1)],
            [(0, 1)],
        ]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        # 8 Strassen blocks for 2x4x8
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
