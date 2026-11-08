import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen decomposition
    # U, V, W are 4x7
    # A, B, C are 2x2. Indices 0,1,2,3 for 00,01,10,11.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify 2x2 Strassen
    # T2[u, v, w] = sum_r U2[u,r] V2[v,r] W2[w,r]
    # u = i*2+j, v = j*2+k, w = k*2+i
    # Check identity: T2[u,v,w] should be 1 if j_u==j_v and i_u==i_w and k_v==k_w
    T2 = np.einsum('ur,vr,wr->uvw', U2, V2, W2)
    # Build target T2
    target_T2 = np.zeros((4, 4, 4))
    for u in range(4):
        i_u, j_u = u // 2, u % 2
        for v in range(4):
            j_v, k_v = v // 2, v % 2
            for w in range(4):
                k_w, i_w = w // 2, w % 2
                if j_u == j_v and i_u == i_w and k_v == k_w:
                    target_T2[u, v, w] = 1
    if not np.allclose(T2, target_T2):
        # Fallback to naive 2x2 if Strassen is wrong (shouldn't happen)
        U2 = np.zeros((4, 8))
        V2 = np.zeros((4, 8))
        W2 = np.zeros((4, 8))
        for r, (i, j, k) in enumerate([(i,j,k) for i in range(2) for j in range(2) for k in range(2)]):
            U2[i*2+j, r] = 1
            V2[j*2+k, r] = 1
            W2[k*2+i, r] = 1
        # Re-verify
        T2 = np.einsum('ur,vr,wr->uvw', U2, V2, W2)
        if not np.allclose(T2, target_T2):
            # Should not happen
            return {"U": [], "V": [], "W": []}

    # 2. Construct 4x4 decomposition from 2x2 blocks
    # Rank = 7 * 7 = 49
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # Map 4x4 indices to 2x2 block indices
    # A 4x4:
    # 00 01 02 03
    # 10 11 12 13
    # 20 21 22 23
    # 30 31 32 33
    # Block 00 (A11): 0,1,4,5
    # Block 01 (A12): 2,3,6,7
    # Block 10 (A21): 8,9,12,13
    # Block 11 (A22): 10,11,14,15
    
    def block_indices_4x4(b_row, b_col):
        # b_row, b_col in {0,1}
        # Returns list of 4 indices in 4x4 matrix (row-major)
        # Block (0,0) is rows 0-1, cols 0-1
        # Indices: r*4 + c
        base_r = b_row * 2
        base_c = b_col * 2
        return [
            base_r*4 + base_c,
            base_r*4 + base_c + 1,
            (base_r+1)*4 + base_c,
            (base_r+1)*4 + base_c + 1
        ]
    
    # Strassen coefficients for 2x2 block multiplication
    # M_k = (sum alpha A_b) (sum beta B_b)
    # C_b = sum delta M_k
    # We need to map the 7 Strassen products to the 4x4 case.
    # Each Strassen product M_k is a 2x2 matrix multiplication.
    # We decompose each M_k using U2, V2, W2.
    # So for each k in 0..6, we have 7 sub-terms r in 0..6.
    # Total 49 terms.
    
    # Strassen alpha, beta, delta for 2x2
    # A blocks: 00, 01, 10, 11
    alpha = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    beta = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    delta = [
        [1, 0, 0, 1], # C11+=M1, C22+=M1
        [0, 0, 1, -1],# C21+=M2, C22-=M2
        [0, 1, 0, 1], # C12+=M3, C22+=M3
        [1, 0, 1, 0], # C11+=M4, C21+=M4
        [-1, 1, 0, 0],# C11-=M5, C12+=M5
        [0, 0, 0, 1], # C22+=M6
        [1, 0, 0, 0]  # C11+=M7
    ]
    
    # Construct 4x4 U, V, W
    for k in range(7):
        for r in range(7):
            idx = k * 7 + r
            
            # U44: 16x1
            u_vec = np.zeros(16)
            # V44: 16x1
            v_vec = np.zeros(16)
            # W44: 16x1
            w_vec = np.zeros(16)
            
            # A contribution
            for b in range(4):
                if alpha[k][b] != 0:
                    b_row, b_col = b // 2, b % 2
                    inds = block_indices_4x4(b_row, b_col)
                    for idx_a, val in zip(inds, U2[:, r]):
                        u_vec[idx_a] += alpha[k][b] * val
                        
            # B contribution
            for b in range(4):
                if beta[k][b] != 0:
                    b_row, b_col = b // 2, b % 2
                    inds = block_indices_4x4(b_row, b_col)
                    for idx_b, val in zip(inds, V2[:, r]):
                        v_vec[idx_b] += beta[k][b] * val
                        
            # C contribution
            for b in range(4):
                if delta[k][b] != 0:
                    b_row, b_col = b // 2, b % 2
                    inds = block_indices_4x4(b_row, b_col)
                    for idx_c, val in zip(inds, W2[:, r]):
                        w_vec[idx_c] += delta[k][b] * val
                        
            U44[:, idx] = u_vec
            V44[:, idx] = v_vec
            W44[:, idx] = w_vec
            
    # Verify 4x4
    T44 = np.einsum('ur,vr,wr->uvw', U44, V44, W44)
    target_T44 = np.zeros((16, 16, 16))
    for u in range(16):
        i_u, j_u = u // 4, u % 4
        for v in range(16):
            j_v, k_v = v // 4, v % 4
            for w in range(16):
                k_w, i_w = w // 4, w % 4
                if j_u == j_v and i_u == i_w and k_v == k_w:
                    target_T44[u, v, w] = 1
    if not np.allclose(T44, target_T44):
        # Fallback to naive 4x4 (rank 64)
        U44 = np.zeros((16, 64))
        V44 = np.zeros((16, 64))
        W44 = np.zeros((16, 64))
        for r, (i, j, k) in enumerate([(i,j,k) for i in range(4) for j in range(4) for k in range(4)]):
            U44[i*4+j, r] = 1
            V44[j*4+k, r] = 1
            W44[k*4+i, r] = 1

    # 3. Construct 4x4 * 4x8
    # B is 4x8. Split into B_L (4x4) and B_R (4x4).
    # C is 4x8. Split into C_L (4x4) and C_R (4x4).
    # Rank = 49 + 49 = 98
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # Part 1: B_L -> C_L
    U_total[:, 0:49] = U44
    # V_L corresponds to B_L (first 16 elements of B flattened)
    V_total[0:16, 0:49] = V44
    # W_L corresponds to C_L (first 16 elements of C flattened? No, W index is k*n+i)
    # C_L is cols 0-3 of C.
    # C is 4x8. C_L is C[:, 0:4].
    # W index for C_L: k in 0..3, i in 0..3.
    # W index = k*4 + i. This is exactly the 16 indices used in W44.
    W_total[0:16, 0:49] = W44
    
    # Part 2: B_R -> C_R
    U_total[:, 49:98] = U44
    # V_R corresponds to B_R (next 16 elements of B flattened)
    V_total[16:32, 49:98] = V44
    # W_R corresponds to C_R (next 16 elements of C flattened? No)
    # C_R is C[:, 4:8].
    # W index for C_R: k in 4..7, i in 0..3.
    # W index = k*4 + i.
    # k=4, i=0 -> 16.
    # k=7, i=3 -> 31.
    # So W_R uses indices 16..31.
    W_total[16:32, 49:98] = W44
    
    # Final Verification
    T_total = np.einsum('ur,vr,wr->uvw', U_total, V_total, W_total)
    target_T_total = np.zeros((16, 32, 32))
    for u in range(16):
        i_u, j_u = u // 4, u % 4
        for v in range(32):
            j_v, k_v = v // 8, v % 8
            for w in range(32):
                k_w, i_w = w // 4, w % 4
                if j_u == j_v and i_u == i_w and k_v == k_w:
                    target_T_total[u, v, w] = 1
                    
    if not np.allclose(T_total, target_T_total):
        # Fallback to naive 4x4*4x8 (rank 128)
        U_total = np.zeros((16, 128))
        V_total = np.zeros((32, 128))
        W_total = np.zeros((32, 128))
        for r, (i, j, k) in enumerate([(i,j,k) for i in range(4) for j in range(4) for k in range(8)]):
            U_total[i*4+j, r] = 1
            V_total[j*8+k, r] = 1
            W_total[k*4+i, r] = 1

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
