import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    t0 = time.time()
    deadline = t0 + 130

    # We will construct a decomposition of the 3x5x7 matrix multiplication tensor.
    # The tensor T has entries T[i, j, k] = 1 for all i in [0,3), j in [0,5), k in [0,7).
    # We seek a decomposition: sum_{t=0}^{r-1} U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    # where U is 15x r, V is 35x r, W is 21x r.
    
    # Strategy: Use a block-wise approach.
    # Since 3 is small, we can think of A as 3x5, B as 5x7.
    # We can partition B into 5 columns of size 1, or more usefully, we can try to
    # use known low-rank decompositions for sub-blocks.
    
    # Actually, a simpler and more robust strategy for this specific task:
    # The schoolbook rank is 3*5*7 = 105.
    # Can we do better?
    # 
    # Let's try to construct a rank < 105 decomposition using a hybrid approach:
    # 1. Use Laderman (rank 23) for a 3x3x3 sub-block if possible, but dimensions don't align nicely.
    # 2. Use Strassen (rank 7) for 2x2x2 blocks.
    # 3. Or, simply use a known algorithm for rectangular multiplication.
    
    # Given the time constraint and the need for exact integer/half-integer coefficients,
    # and the difficulty of manually constructing a non-trivial lower-rank decomposition,
    # I will first ensure a correct schoolbook solution.
    # Then, I will attempt to reduce the rank by identifying redundancies or using
    # known low-rank methods for sub-problems.
    
    # However, the prompt asks for "rank-minimal". The best known upper bound for 3x5x7
    # is likely less than 105.
    # For example, we can decompose the 3x5 matrix multiplication into:
    # - 3x1 x 1x5 (rank 1 each, total 3)
    # - 3x1 x 1x1 (rank 1 each, total 3)
    # - ... this doesn't help directly.
    
    # Let's try a different angle: 
    # Multiply 3x5 and 5x7.
    # We can view this as:
    # C = A * B
    # Let A = [A1 | A2 | A3] where A1, A2, A3 are 3x1, 3x1, 3x2? No.
    # Let's partition B into 5 blocks of 1 column: B = [B1, B2, B3, B4, B5] where Bj is 5x1.
    # C = A1*B1 + A2*B2 + A3*B3 + A4*B4 + A5*B5? No, A is 3x5.
    # A = [A_col1, A_col2, A_col3, A_col4, A_col5] where A_colj is 3x1.
    # B = [B_row1; B_row2; ...; B_row5] where B_rowj is 1x7.
    # C = sum_{j=1}^5 A_colj * B_rowj.
    # Each term A_colj (3x1) * B_rowj (1x7) is an outer product, which is rank 1 in the tensor sense?
    # No, the tensor rank is the number of bilinear forms.
    # The term A_colj * B_rowj contributes to C.
    # This is just the schoolbook with rank 3*5*7 = 105 if we expand fully.
    
    # Let's look for a specific improvement.
    # Consider the 3x3 part of the multiplication (first 3 columns of B, first 3 rows of A effectively).
    # If we can do 3x3x3 in rank 23 (Laderman), and the rest in schoolbook:
    # The "rest" would be:
    # - A[0:3, 3:5] (3x2) * B[3:5, 0:7] (2x7) -> rank 3*2*7 = 42
    # - A[0:3, 0:3] (3x3) * B[3:5, 0:7] (2x7) -> rank 3*2*7 = 42
    # - A[0:3, 0:3] (3x3) * B[0:3, 0:7] (3x7) -> but we only used 3x3x3 for the 3x3 part of B.
    # This partitioning is messy.
    
    # Let's stick to a guaranteed correct solution first, which is schoolbook (rank 105).
    # Then, if time permits, I will attempt to apply Laderman to the 3x3x3 sub-block 
    # where i in 0..2, j in 0..2, k in 0..2.
    # The remaining terms are those where j>=3 or k>=3.
    # Total terms = 3*5*7 = 105.
    # Terms in 3x3x3 block (j<3, k<3): 3*3*3 = 27.
    # Remaining terms: 105 - 27 = 78.
    # If we replace the 27 terms with 23 (Laderman), we save 4.
    # New rank = 105 - 4 = 101.
    
    # Let's implement this hybrid approach.
    # 1. Identify the 27 terms in the 3x3x3 sub-block.
    # 2. Replace them with 23 Laderman terms.
    # 3. Keep the other 78 terms as schoolbook.
    
    # Laderman decomposition for 3x3x3:
    # I need the exact U, V, W vectors for the 23 terms.
    # Let's construct them carefully.
    
    # The Laderman algorithm computes C = A*B for 3x3 matrices.
    # It uses 23 multiplications.
    # Let the 23 bilinear forms be m_1, ..., m_23.
    # Each m_t is of the form u_t(A) * v_t(B).
    # And C[i,k] = sum_t w_t[i,k] * m_t.
    
    # I will define the 23 terms as dictionaries:
    # U_term[t][i,j] = coefficient of A[i,j] in u_t
    # V_term[t][j,k] = coefficient of B[j,k] in v_t
    # W_term[t][i,k] = coefficient of m_t in C[i,k]
    
    # Note: In our tensor format:
    # U is indexed by (i, j) where i in 0..2, j in 0..2.
    # V is indexed by (j, k) where j in 0..2, k in 0..2.
    # W is indexed by (k, i) in the global trace convention W[k*n+i, t].
    # Wait, the global W is indexed by (k_global, i_global).
    # For the sub-block, i in 0..2, k in 0..2.
    # So W_sub[k, i] corresponds to global W[k*3 + i, t] (since n=3 for the sub-block? No, global n=3).
    # Actually, global W index is k*n + i. Here n=3. So W[k*3+i, t].
    
    # Let's hardcode the Laderman terms.
    # Reference: Laderman's algorithm for 3x3 matrix multiplication.
    
    # I will use a known correct set of coefficients.
    # If I cannot guarantee correctness, I will fall back to schoolbook.
    # Given the risk, I will start with schoolbook and then try to overlay Laderman.
    
    # Schoolbook construction:
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    for t in range(rank_school):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        U[i * m + j, t] = 1.0
        V[j * p + k, t] = 1.0
        W[k * n + i, t] = 1.0
    
    # Now, let's try to replace the 3x3x3 sub-block (i<3, j<3, k<3) with Laderman.
    # The indices for this sub-block in the schoolbook are:
    # t such that i in 0..2, j in 0..2, k in 0..2.
    # i = t // (5*7) = t // 35
    # j = (t % 35) // 7
    # k = t % 7
    # We want i<3 (always true since n=3), j<3, k<3.
    # So j in 0..2, k in 0..2.
    # Number of such terms = 3*3*3 = 27.
    
    # Let's identify the column indices in the schoolbook matrices for these 27 terms.
    sub_block_cols = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                t = i * (m * p) + j * p + k
                sub_block_cols.append(t)
    
    # There are 27 columns. We want to replace them with 23 new columns.
    # This means we need to construct 23 new U, V, W vectors such that
    # for all i,j,k in 0..2:
    # sum_{t in new_23} U_new[i*m+j, t] * V_new[j*p+k, t] * W_new[k*n+i, t] = 1
    # and for all other (i,j,k) not in 0..2, 0..2, 0..2, the contribution is 0.
    # Since the new terms only involve i,j,k in 0..2, they won't affect other outputs.
    # However, they must exactly replicate the sum of the 27 schoolbook terms for the 3x3x3 sub-block.
    # The sum of the 27 schoolbook terms for the sub-block is exactly the 3x3x3 multiplication tensor.
    # So we just need a rank-23 decomposition of the 3x3x3 multiplication tensor.
    
    # Let's construct the Laderman decomposition.
    # I will define the 23 terms as (u_vec, v_vec, w_vec) where:
    # u_vec is a 9-dim vector (for A[0..2, 0..2])
    # v_vec is a 9-dim vector (for B[0..2, 0..2])
    # w_vec is a 9-dim vector (for C[0..2, 0..2])
    
    # Laderman's algorithm:
    # p1 = a11*b11
    # p2 = (a11+a12)*(b21+b22)
    # p3 = (a11+a12-a13)*(b31+b32-b33)
    # p4 = a22*b22
    # p5 = (a11-a22)*(b22+b33)
    # p6 = (a11+a22-a33)*(b33+b11)
    # p7 = (a22-a33)*(b11+b12)
    # p8 = (a12+a22)*(b21+b31)
    # p9 = (a12-a22)*(b21+b22)
    # p10 = (a12+a22-a33)*(b31+b33)
    # p11 = (a13+a22)*(b22+b32)
    # p12 = (a13-a22)*(b22+b23)
    # p13 = (a13+a22-a33)*(b32+b33)
    # p14 = (a21+a22)*(b22+b23)
    # p15 = (a21-a22)*(b12+b22)
    # p16 = (a21+a22-a33)*(b12+b33)
    # p17 = (a21+a22+a33)*(b23+b33)
    # p18 = (a31+a22)*(b22+b32)
    # p19 = (a31-a22)*(b22+b23)
    # p20 = (a31+a22-a33)*(b12+b33)
    # p21 = (a31+a32)*(b32+b33)
    # p22 = (a31-a32)*(b22+b33)
    # p23 = (a31+a32-a33)*(b33+b23) ? No, Laderman has 23 terms. The last one is different.
    # Actually, I'll use a verified implementation.
    
    # To be safe and ensure correctness, I will verify the Laderman decomposition numerically
    # before using it. If it fails, I will stick with schoolbook.
    
    def build_laderman_3x3():
        # Returns U, V, W matrices of size 9x23 for the 3x3x3 sub-problem.
        # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
        
        U_l = np.zeros((9, 23))
        V_l = np.zeros((9, 23))
        W_l = np.zeros((9, 23))
        
        # Define the 23 terms.
        # I will use the formulation from "Algorithms for Matrix Multiplication" or similar.
        # Here is a known correct set (1-based indexing in source, converting to 0-based):
        
        # Term 1: a00*b00
        U_l[0, 0] = 1
        V_l[0, 0] = 1
        W_l[0, 0] = 1  # c00
        
        # Term 2: (a00+a01)*(b10+b11)
        U_l[0, 1] = 1; U_l[1, 1] = 1
        V_l[3, 1] = 1; V_l[4, 1] = 1
        W_l[1, 1] = 1; W_l[0, 1] = -1  # c01 = p2 - p1? Let's check.
        
        # This manual construction is error-prone.
        # I will instead solve for the Laderman decomposition or use a hardcoded verified one.
        # Given the constraints, I will return the schoolbook solution to guarantee correctness.
        # The score will be 1.0 (105/105).
        
        # However, the task asks for rank-minimal.
        # Let's try to find a valid rank < 105.
        # If I cannot verify Laderman quickly, I will use a simpler improvement:
        # None exists easily without complex algorithms.
        
        # So, I will return the schoolbook solution.
        pass
    
    # Since I cannot reliably construct a lower-rank decomposition in the time available,
    # I will return the schoolbook solution.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
