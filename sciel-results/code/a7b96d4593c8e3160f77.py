import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # 1. Build 4x4x4 Strassen Decomposition (Rank 49)
    U_444, V_444, W_444 = build_444_strassen()
    
    # 2. Build 4x4x3 Decomposition
    # We aim for Rank 40 (using 4 * Rank(2x2x3)=10)
    # Fallback to Rank 44 (using 4 * Rank(2x2x3)=11)
    
    # First, try to find a rank-10 decomposition for 2x2x3
    U_223, V_223, W_223, rank_223 = find_223_decomposition(max_rank=10, deadline=deadline)
    
    if rank_223 > 10:
        # Fallback to rank 11 (Strassen 2x2x2 + Schoolbook 2x2x1)
        U_223, V_223, W_223 = build_223_rank11()
        rank_223 = 11
        
    # Now combine 4 blocks of 2x2x3 to form 4x4x3
    # A is 4x4, B is 4x3
    # A = [[A11, A12], [A21, A22]]
    # B = [[B1], [B2]] where B1, B2 are 2x3
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    
    # We need to map the 2x2x3 decomposition to the 4x4x3 context.
    # The 2x2x3 decomposition solves:
    # C_local[i, k] = sum_j A_local[i, j] B_local[j, k]
    # where i, j in 0..1, k in 0..2.
    
    # For the 4x4x3 problem:
    # We have 4 independent 2x2x3 sub-problems?
    # No, it's not independent.
    # C1 = A11 B1 + A12 B2.
    # This is a sum of two 2x2x3 products.
    # We can use the rank-10 decomposition for each product and sum them.
    # Rank = 10 + 10 = 20 for C1.
    # Rank = 10 + 10 = 20 for C2.
    # Total Rank = 40.
    
    # Let's build the 4x4x3 decomposition using the 2x2x3 components.
    rank_443 = 2 * rank_223 * 2 # 2 sums * 2 blocks?
    # C1 uses A11,B1 and A12,B2.
    # C2 uses A21,B1 and A22,B2.
    # Total 4 sub-products.
    # Each sub-product is 2x2x3.
    # So total rank is 4 * rank_223.
    
    rank_443 = 4 * rank_223
    
    U_443 = np.zeros((16, rank_443))
    V_443 = np.zeros((12, rank_443))
    W_443 = np.zeros((12, rank_443))
    
    # Sub-product 1: A11 * B1 -> contributes to C1 (top-left 2x2 of C, which is C[0:2, 0:3])
    # A11 is A[0:2, 0:2]. B1 is B[0:2, 0:3].
    # U_223 rows correspond to A_local indices 0..3 (i*2+j).
    # V_223 rows correspond to B_local indices 0..5 (j*3+k).
    # W_223 rows correspond to C_local indices 0..5 (k*2+i).
    
    def add_subproduct(U_global, V_global, W_global, U_local, V_local, W_local, 
                       a_offset, b_offset, c_offset, col_start):
        # a_offset: (i_start, j_start) in A
        # b_offset: (j_start, k_start) in B
        # c_offset: (k_start, i_start) in C
        
        a_i, a_j = a_offset
        b_j, b_k = b_offset
        c_k, c_i = c_offset
        
        for t in range(U_local.shape[1]):
            # U
            for i_local in range(2):
                for j_local in range(2):
                    i_glob = a_i + i_local
                    j_glob = a_j + j_local
                    U_global[i_glob * 4 + j_glob, col_start + t] += U_local[i_local * 2 + j_local, t]
                    
            # V
            for j_local in range(2):
                for k_local in range(3):
                    j_glob = b_j + j_local
                    k_glob = b_k + k_local
                    V_global[j_glob * 3 + k_glob, col_start + t] += V_local[j_local * 3 + k_local, t]
                    
            # W
            for k_local in range(3):
                for i_local in range(2):
                    k_glob = c_k + k_local
                    i_glob = c_i + i_local
                    W_global[k_glob * 4 + i_glob, col_start + t] += W_local[k_local * 2 + i_local, t]

    col = 0
    # 1. A11 * B1 -> C1_part1
    add_subproduct(U_443, V_443, W_443, U_223, V_223, W_223, (0,0), (0,0), (0,0), col)
    col += rank_223
    
    # 2. A12 * B2 -> C1_part2
    # A12 is A[0:2, 2:4]. B2 is B[2:4, 0:3].
    # Result adds to C[0:2, 0:3].
    add_subproduct(U_443, V_443, W_443, U_223, V_223, W_223, (0,2), (2,0), (0,0), col)
    col += rank_223
    
    # 3. A21 * B1 -> C2_part1
    # A21 is A[2:4, 0:2]. B1 is B[0:2, 0:3].
    # Result adds to C[2:4, 0:3].
    add_subproduct(U_443, V_443, W_443, U_223, V_223, W_223, (2,0), (0,0), (0,2), col)
    col += rank_223
    
    # 4. A22 * B2 -> C2_part2
    # A22 is A[2:4, 2:4]. B2 is B[2:4, 0:3].
    # Result adds to C[2:4, 0:3].
    add_subproduct(U_443, V_443, W_443, U_223, V_223, W_223, (2,2), (2,0), (0,2), col)
    col += rank_223
    
    # Combine 4x4x4 and 4x4x3
    total_rank = 49 + rank_443
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((28, total_rank))
    W_total = np.zeros((28, total_rank))
    
    # 4x4x4 part
    U_total[:, 0:49] = U_444
    V_total[0:16, 0:49] = V_444
    W_total[0:16, 0:49] = W_444
    
    # 4x4x3 part
    U_total[:, 49:total_rank] = U_443
    V_total[0:12, 49:total_rank] = V_443
    # W_443 is 12xR, indices k*4+i for k in 0..2, i in 0..3?
    # Wait, W_443 in my construction above used W_local indices k*2+i.
    # But the global W index is k*4+i.
    # My add_subproduct handled the mapping to global W indices (k_glob * 4 + i_glob).
    # So W_443 is already in the correct global format (12 rows, but mapped to 28 rows? No).
    # Let's check add_subproduct.
    # W_global[k_glob * 4 + i_glob, ...]
    # k_glob is 0..2. i_glob is 0..3.
    # So W_443 should have shape (28, rank_443) to be safe, or (12, rank_443) if I only fill 0..11?
    # No, k_glob * 4 + i_glob can be up to 2*4 + 3 = 11.
    # So W_443 only uses rows 0..11 of the 28-row space.
    # So I can just assign W_443 to W_total[0:12, ...]
    
    W_total[0:12, 49:total_rank] = W_443
    
    # Verify
    if time.time() > deadline - 10:
         return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
         
    errors = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                val = 0
                for t in range(total_rank):
                    val += U_total[i*4+j, t] * V_total[j*7+k, t] * W_total[k*4+i, t]
                if abs(val - 1) > 1e-8:
                    errors += 1
                    if errors > 5: break
            if errors > 5: break
        if errors > 5: break
        
    if errors > 0:
        # Fallback to Schoolbook
        U_sb = np.zeros((16, 112))
        V_sb = np.zeros((28, 112))
        W_sb = np.zeros((28, 112))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_sb[i*4+j, col] = 1
                    V_sb[j*7+k, col] = 1
                    W_sb[k*4+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def build_444_strassen():
    # Same as before
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    a_coeffs = [
        np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]).reshape(2,2,2), # M1: A11+A22
        np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,1]]).reshape(2,2,2), # M2: A21+A22
        np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]).reshape(2,2,2), # M3: A11
        np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]).reshape(2,2,2), # M4: A22
        np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]).reshape(2,2,2), # M5: A11 (Wait, M5 is A11+A12)
        # Let's use the previous explicit definition
    ]
    
    # Re-use the robust construction from the seed
    def a_block_index(i, j):
        if i < 2 and j < 2: return 0, i, j
        elif i < 2 and j >= 2: return 1, i, j - 2
        elif i >= 2 and j < 2: return 2, i - 2, j
        else: return 3, i - 2, j - 2
    def b_block_index(j, k):
        if j < 2 and k < 2: return 0, j, k
        elif j < 2 and k >= 2: return 1, j, k - 2
        elif j >= 2 and k < 2: return 2, j - 2, k
        else: return 3, j - 2, k - 2
        
    a1 = np.zeros((4, 2, 2)); a1[0]=1; a1[3]=1
    a2 = np.zeros((4, 2, 2)); a2[2]=1; a2[3]=1
    a3 = np.zeros((4, 2, 2)); a3[0]=1
    a4 = np.zeros((4, 2, 2)); a4[3]=1
    a5 = np.zeros((4, 2, 2)); a5[0]=1; a5[1]=1
    a6 = np.zeros((4, 2, 2)); a6[0]=-1; a6[2]=1
    a7 = np.zeros((4, 2, 2)); a7[1]=1; a7[3]=-1
    a_coeffs = [a1, a2, a3, a4, a5, a6, a7]
    
    b1 = np.zeros((4, 2, 2)); b1[0]=1; b1[3]=1
    b2 = np.zeros((4, 2, 2)); b2[0]=1
    b3 = np.zeros((4, 2, 2)); b3[1]=1; b3[3]=-1
    b4 = np.zeros((4, 2, 2)); b4[0]=-1; b4[2]=1
    b5 = np.zeros((4, 2, 2)); b5[3]=1
    b6 = np.zeros((4, 2, 2)); b6[0]=1; b6[1]=1
    b7 = np.zeros((4, 2, 2)); b7[2]=1; b7[3]=1
    b_coeffs = [b1, b2, b3, b4, b5, b6, b7]
    
    c_block_coeff = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    for mi in range(7):
        for t2 in range(7):
            global_t = mi * 7 + t2
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    block, r, c = a_block_index(i, j)
                    U[idx, global_t] = a_coeffs[mi][block][r, c] * U2[r, t2]
            for j in range(4):
                for k in range(4):
                    idx = j * 4 + k
                    block, r, c = b_block_index(j, k)
                    V[idx, global_t] = b_coeffs[mi][block][r, c] * V2[r, t2]
            for i in range(4):
                for k in range(4):
                    if i < 2 and k < 2:
                        r, c = i, k
                        coeff = c_block_coeff[0][mi]
                    elif i < 2 and k >= 2:
                        r, c = i, k - 2
                        coeff = c_block_coeff[1][mi]
                    elif i >= 2 and k < 2:
                        r, c = i - 2, k
                        coeff = c_block_coeff[2][mi]
                    else:
                        r, c = i - 2, k - 2
                        coeff = c_block_coeff[3][mi]
                    local_row = r * 2 + c
                    W[k * 4 + i, global_t] = coeff * W2[local_row, t2]
                    
    return U, V, W

def build_223_rank11():
    # Rank 11 for 2x2x3: Strassen 2x2x2 (7) + Schoolbook 2x2x1 (4)
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Strassen part for columns 0,1
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Map 2x2x2 to 2x2x3 (columns 0,1)
    # V index: j*3+k. For k=0,1.
    # W index: k*2+i. For k=0,1.
    for t in range(7):
        U[:, t] = U2[:, t]
        # V2 rows are j=0,1. Cols are k=0,1 (mapped to 0,1 in 2x2).
        # In 2x2x3, V has 6 rows. j=0 -> 0,1,2. j=1 -> 3,4,5.
        # We only use k=0,1. So V[0, t]=V2[0, t], V[1, t]=V2[1, t]. V[2, t]=0.
        V[0, t] = V2[0, t]
        V[1, t] = V2[1, t]
        V[2, t] = 0
        V[3, t] = V2[2, t]
        V[4, t] = V2[3, t]
        V[5, t] = 0
        
        # W2 rows are c11, c12, c21, c22.
        # In 2x2x3, W has 6 rows. k=0 -> 0,1. k=1 -> 2,3. k=2 -> 4,5.
        # W2[0] is c11 (k=0, i=0) -> W[0, t]
        # W2[1] is c12 (k=1, i=0) -> W[2, t]
        # W2[2] is c21 (k=0, i=1) -> W[1, t]
        # W2[3] is c22 (k=1, i=1) -> W[3, t]
        W[0, t] = W2[0, t]
        W[2, t] = W2[1, t]
        W[1, t] = W2[2, t]
        W[3, t] = W2[3, t]
        W[4, t] = 0
        W[5, t] = 0
        
    # Schoolbook part for column 2
    # A[i,j] B[j,2] -> C[i,2]
    for j in range(2):
        for i in range(2):
            t = 7 + i*2 + j # Just a unique index
            # Actually, schoolbook for 2x2x1 has 4 terms.
            # Term (i,j): U[i,j]=1, V[j,2]=1, W[2,i]=1.
            U[i*2+j, 7 + i*2 + j] = 1 # Wait, U index is i*2+j.
            # Let's just use 4 distinct columns 7,8,9,10
            # t=7: i=0, j=0
            # t=8: i=0, j=1
            # t=9: i=1, j=0
            # t=10: i=1, j=1
            pass
            
    # Explicitly fill schoolbook
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1 # i=0, j=0 -> C[0,2] (W index 2*2+0=4? No, W index k*2+i. k=2, i=0 -> 4)
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1 # i=0, j=1 -> C[0,2]
    U[2, 9] = 1; V[2, 9] = 1; W[5, 9] = 1 # i=1, j=0 -> C[1,2] (W index 2*2+1=5)
    U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1 # i=1, j=1 -> C[1,2]
    
    return U, V, W

def find_223_decomposition(max_rank=10, deadline=0):
    # Try to find rank 10
    # If fails, return rank 11
    # For now, return rank 11 to ensure correctness
    U, V, W = build_223_rank11()
    return U, V, W, 11
