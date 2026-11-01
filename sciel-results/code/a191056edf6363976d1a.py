"""
Strategy: Direct Sum Decomposition with Optimized 2x8 Block.
We decompose the 2x2x15 tensor into a direct sum of a 2x2x8 tensor and a 2x2x7 tensor.
1. The 2x2x8 tensor is decomposed using the tensor power of the Strassen algorithm (rank 7^3 = 343? No, that's for 8x8x8).
   For 2x2x8, we can view it as 4 blocks of 2x2x2.
   The rank of M_{2,2,8} is known to be 32 (4 * 8) via schoolbook, but can it be lower?
   Actually, the rank of M_{2,2,2^k} is R_{2,2,2^k}.
   R_{2,2,2} = 7.
   R_{2,2,4} = 7^2 = 49? No, the rank of the tensor power is the product of ranks.
   So R_{2,2,4} <= 49. Schoolbook is 16. So 16 is better.
   Wait, Strassen is for 2x2x2.
   For 2x2x4, we can use 2 Strassen blocks?
   The tensor M_{2,2,4} is the direct sum of two M_{2,2,2} tensors.
   So rank(M_{2,2,4}) <= 7 + 7 = 14.
   Schoolbook rank is 16. So 14 is an improvement.
   
   For 2x2x8, it is the direct sum of four M_{2,2,2} tensors.
   Rank <= 4 * 7 = 28.
   Schoolbook rank is 32. So 28 is an improvement.
   
   For 2x2x7, it is the direct sum of three M_{2,2,2} tensors and one M_{2,2,1} tensor.
   Rank <= 3 * 7 + 2 = 23.
   Schoolbook rank is 28. So 23 is an improvement.
   
   Total rank = 28 + 23 = 51.
   This is better than the current program's 53 (which used 7 + 13*2 = 33? Wait.
   The current program used 7 (for first 2) + 13 * 2 (for next 13) = 7 + 26 = 33.
   Wait, the current program's score is 1.0, which implies rank 60?
   No, the score is seed/value. Seed is 60.
   If value is 53, score is 60/53 = 1.132.
   The current program in the prompt has score 1.0000.
   Let's re-read the current program.
   It constructs a hybrid: Strassen for k=0,1 (rank 7) and Rank 2 for k=2..14 (13 * 2 = 26).
   Total rank = 33.
   Why is the score 1.0000?
   Maybe the verification failed and it fell back to schoolbook (rank 60)?
   Let's check the verification logic in the current program.
   `verify(U, V, W)` checks if the tensor is all 1s.
   The construction seems correct.
   Why would it fail?
   Maybe the indices in `_get_strassen_222` are wrong?
   Or maybe `_get_rank3_221` is wrong?
   Let's re-verify the Rank 2 for 2x2x1.
   U = [[1,0],[0,1],[1,0],[0,1]]
   V = [[1,0],[0,1]]
   W = [[1,1],[1,1]]
   
   Term 0:
   U[:,0] = [1,0,1,0] -> a11, a21
   V[:,0] = [1,0] -> b1
   W[:,0] = [1,1] -> c1, c2
   Contribution:
   T[0,0,0] = 1*1*1 = 1 (a11 b1 -> c1). Correct.
   T[1,0,0] = 1*1*1 = 1 (a21 b1 -> c2). Correct.
   T[0,1,0] = 0. Correct.
   T[1,1,0] = 0. Correct.
   
   Term 1:
   U[:,1] = [0,1,0,1] -> a12, a22
   V[:,1] = [0,1] -> b2
   W[:,1] = [1,1] -> c1, c2
   Contribution:
   T[0,1,0] = 1*1*1 = 1 (a12 b2 -> c1). Correct.
   T[1,1,0] = 1*1*1 = 1 (a22 b2 -> c2). Correct.
   T[0,0,0] = 0. Correct.
   T[1,0,0] = 0. Correct.
   
   So Rank 2 for 2x2x1 is correct.
   
   Now Strassen for 2x2x2.
   Let's check the indices in `_get_strassen_222`.
   U, V, W are 4x7.
   The verification uses `np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)`.
   U_r is (2,2,7). V_r is (2,2,7). W_r is (2,2,7).
   This computes T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
   Wait. The tensor for matrix multiplication C = A B is T[i,j,k] = 1 if C[i,k] = A[i,j] B[j,k].
   The indices are:
   i: row of A, row of C.
   j: col of A, row of B.
   k: col of B, col of C.
   
   In the factorization:
   U has indices (i, j, c).
   V has indices (j, k, c).
   W has indices (k, i, c).
   
   The product is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
   This matches T[i,j,k].
   
   So the verification is correct.
   Why did the current program score 1.0?
   Maybe the Strassen coefficients are wrong?
   Let's check Strassen coefficients.
   M1 = (a11+a22)(b11+b22)
   M2 = (a21+a22)b11
   M3 = a11(b12-b22)
   M4 = a22(b21-b11)
   M5 = (a11+a12)b22
   M6 = (a21-a11)(b11+b12)
   M7 = (a12-a22)(b21+b22)
   
   C11 = M1 - M3 - M4 + M6
   C12 = M3 + M5
   C21 = M2 + M4
   C22 = M1 - M2 - M5 + M7
   
   Let's check W in the current program.
   W is (4, 7). W[k*n+i, c].
   n=2.
   W[0, :] -> i=0, k=0 -> C11.
   W[1, :] -> i=1, k=0 -> C21.
   W[2, :] -> i=0, k=1 -> C12.
   W[3, :] -> i=1, k=1 -> C22.
   
   Current code:
   W[0, 0] = 1; W[0, 2] = -1; W[0, 3] = -1; W[0, 5] = 1
   -> C11 = M1 - M3 - M4 + M6. Correct.
   
   W[1, 2] = 1; W[1, 4] = 1
   -> C21 = M3 + M5. 
   Wait. C21 = M2 + M4.
   Here W[1, :] is for i=1, k=0 -> C21.
   The code sets W[1, 2] = 1 (M3) and W[1, 4] = 1 (M5).
   This is WRONG. C21 should be M2 + M4.
   M2 is index 1. M4 is index 3.
   So W[1, 1] = 1; W[1, 3] = 1.
   
   Let's check the code again.
   `W[1, 2] = 1; W[1, 4] = 1`
   This corresponds to C21 = M3 + M5.
   But C12 = M3 + M5.
   So the code swapped C21 and C12?
   Let's check W[2, :].
   W[2, 1] = 1; W[2, 3] = 1
   -> C12 = M2 + M4.
   But C12 should be M3 + M5.
   
   So the W matrix in the current program is incorrect.
   Specifically, rows 1 and 2 are swapped or misassigned.
   W[1, :] is for C21 (i=1, k=0).
   W[2, :] is for C12 (i=0, k=1).
   
   Correct assignments:
   C11 (i=0, k=0, idx 0): M1 - M3 - M4 + M6 -> W[0, 0]=1, W[0, 2]=-1, W[0, 3]=-1, W[0, 5]=1.
   C12 (i=0, k=1, idx 2): M3 + M5 -> W[2, 2]=1, W[2, 4]=1.
   C21 (i=1, k=0, idx 1): M2 + M4 -> W[1, 1]=1, W[1, 3]=1.
   C22 (i=1, k=1, idx 3): M1 - M2 - M5 + M7 -> W[3, 0]=1, W[3, 1]=-1, W[3, 4]=-1, W[3, 6]=1.
   
   The current program has:
   W[1, 2] = 1; W[1, 4] = 1  (This is M3+M5, which is C12, but placed at idx 1 which is C21)
   W[2, 1] = 1; W[2, 3] = 1  (This is M2+M4, which is C21, but placed at idx 2 which is C12)
   
   So the current program fails verification and falls back to schoolbook rank 60.
   Score = 60/60 = 1.0.
   
   My new strategy will fix the Strassen coefficients and use the direct sum decomposition.
   Rank 51 is expected.
   Score = 60/51 = 1.176.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Corrected Strassen decomposition.
    """
    U = np.zeros((4, 7))
    # a11:0, a12:1, a21:2, a22:3
    # M1: a11+a22
    U[0, 0] = 1; U[3, 0] = 1
    # M2: a21+a22
    U[2, 1] = 1; U[3, 1] = 1
    # M3: a11
    U[0, 2] = 1
    # M4: a22
    U[3, 3] = 1
    # M5: a11+a12
    U[0, 4] = 1; U[1, 4] = 1
    # M6: a21-a11
    U[2, 5] = 1; U[0, 5] = -1
    # M7: a12-a22
    U[1, 6] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7))
    # b11:0, b12:1, b21:2, b22:3
    # M1: b11+b22
    V[0, 0] = 1; V[3, 0] = 1
    # M2: b11
    V[0, 1] = 1
    # M3: b12-b22
    V[1, 2] = 1; V[3, 2] = -1
    # M4: b21-b11
    V[2, 3] = 1; V[0, 3] = -1
    # M5: b22
    V[3, 4] = 1
    # M6: b11+b12
    V[0, 5] = 1; V[1, 5] = 1
    # M7: b21+b22
    V[2, 6] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7))
    # W index: k*n + i. n=2.
    # idx 0: i=0, k=0 (C11)
    # idx 1: i=1, k=0 (C21)
    # idx 2: i=0, k=1 (C12)
    # idx 3: i=1, k=1 (C22)
    
    # C11 = M1 - M3 - M4 + M6
    W[0, 0] = 1; W[0, 2] = -1; W[0, 3] = -1; W[0, 5] = 1
    
    # C21 = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C12 = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C22 = M1 - M2 - M5 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 4] = -1; W[3, 6] = 1
    
    return U, V, W

def _get_rank2_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 2.
    """
    U = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    V = np.array([
        [1, 0],
        [0, 1]
    ])
    W = np.array([
        [1, 1],
        [1, 1]
    ])
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_str, V_str, W_str = _get_strassen_222()
    U_r2, V_r2, W_r2 = _get_rank2_221()
    
    # Decompose p=15 into 4 blocks of 2 and 1 block of 1?
    # 15 = 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1?
    # 4 * 2 = 8.
    # 15 - 8 = 7.
    # 7 = 2 + 2 + 2 + 1.
    # So 7 blocks of 2 and 1 block of 1.
    # Total blocks of 2: 7.
    # Total blocks of 1: 1.
    # Rank = 7 * 7 + 1 * 2 = 49 + 2 = 51.
    
    num_blocks_2 = 7
    num_blocks_1 = 1
    
    R_total = num_blocks_2 * 7 + num_blocks_1 * 2
    
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    col_offset = 0
    k_offset = 0
    
    # Fill 7 blocks of 2x2x2
    for _ in range(num_blocks_2):
        # U
        U[:, col_offset:col_offset+7] = U_str
        
        # V
        # V indices for this block: j*p + k_offset .. j*p + k_offset+1
        for j in range(m):
            V[j*p + k_offset, col_offset:col_offset+7] = V_str[j, :]
            V[j*p + k_offset + 1, col_offset:col_offset+7] = V_str[j+1, :]
            # Wait, V_str is 4x7.
            # V_str row 0: j=0, k=0
            # V_str row 1: j=0, k=1
            # V_str row 2: j=1, k=0
            # V_str row 3: j=1, k=1
            # So for global k_offset:
            # V[j*p + k_offset, col] = V_str[2*j, col]
            # V[j*p + k_offset+1, col] = V_str[2*j+1, col]
            
            V[j*p + k_offset, col_offset:col_offset+7] = V_str[2*j, :]
            V[j*p + k_offset + 1, col_offset:col_offset+7] = V_str[2*j+1, :]
            
        # W
        # W indices for this block: (k_offset)*n + i .. (k_offset+1)*n + i
        for i in range(n):
            W[k_offset*n + i, col_offset:col_offset+7] = W_str[2*i, :]
            W[(k_offset+1)*n + i, col_offset:col_offset+7] = W_str[2*i+1, :]
            # W_str row 0: i=0, k=0
            # W_str row 1: i=1, k=0
            # W_str row 2: i=0, k=1
            # W_str row 3: i=1, k=1
            # So for global k_offset:
            # W[k_offset*n + i, col] = W_str[2*i, col]
            # W[(k_offset+1)*n + i, col] = W_str[2*i+1, col]
            
        col_offset += 7
        k_offset += 2
        
    # Fill 1 block of 2x2x1
    for _ in range(num_blocks_1):
        # U
        U[:, col_offset:col_offset+2] = U_r2
        
        # V
        for j in range(m):
            V[j*p + k_offset, col_offset:col_offset+2] = V_r2[j, :]
            
        # W
        for i in range(n):
            W[k_offset*n + i, col_offset:col_offset+2] = W_r2[i, :]
            
        col_offset += 2
        k_offset += 1
        
    # Verify
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
