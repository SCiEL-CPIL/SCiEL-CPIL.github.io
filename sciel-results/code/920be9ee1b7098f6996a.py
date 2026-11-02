"""
Matrix multiplication 2x4 x 4x7 with rank-minimal exact tensor decomposition.
Uses Strassen's algorithm for the 2x2x2 sub-problems to reduce rank from 56 (schoolbook) to 50.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition using Strassen for 2x2x2 blocks.
    # The middle dimension m=4 can be split into two blocks of size 2.
    # For each block, we have a 2x2 (A) times 2x7 (B) multiplication.
    # We further split the 7 columns of B into three 2-column blocks and one 1-column block.
    # So we have 2 (middle blocks) * 4 (p-subblocks) = 8 sub-problems.
    # Each 2x2x2 sub-problem uses Strassen (rank 7).
    # The 2x2x1 sub-problems use schoolbook (rank 2).
    
    # Total rank = 2 * (3*7 + 1*2) = 2 * (21 + 2) = 2 * 23 = 46? 
    # Wait, let me recalculate.
    
    # For each middle block (2 of them):
    #   A is 2x2, B is 2x7.
    #   We split B into columns [0,1], [2,3], [4,5], [6].
    #   Sub-problems:
    #     1. A(2x2) * B_cols(2x2) -> 2x2 result. Strassen: rank 7.
    #     2. A(2x2) * B_cols(2x2) -> 2x2 result. Strassen: rank 7.
    #     3. A(2x2) * B_cols(2x2) -> 2x2 result. Strassen: rank 7.
    #     4. A(2x2) * B_col(2x1) -> 2x1 result. Schoolbook: rank 2.
    #   Total per middle block: 7+7+7+2 = 23.
    #   Two middle blocks: 23 * 2 = 46.
    
    # But wait, the output C is 2x7. The sub-problems produce parts of C.
    # For middle block b (b=0,1):
    #   A_cols = A[:, b*2 : (b+1)*2]  (2x2)
    #   B_rows = B[b*2 : (b+1)*2, :]  (2x7)
    #   Contribution to C = A_cols @ B_rows (2x7)
    #   
    #   We split B_rows into 4 sub-matrices:
    #     B_rows[:, 0:2] (2x2) -> C_part1 (2x2)
    #     B_rows[:, 2:4] (2x2) -> C_part2 (2x2)
    #     B_rows[:, 4:6] (2x2) -> C_part3 (2x2)
    #     B_rows[:, 6:7] (2x1) -> C_part4 (2x1)
    #   
    #   C = sum over b of (A_cols_b @ B_rows_b)
    #     = sum over b of [C_part1_b, C_part2_b, C_part3_b, C_part4_b]
    #   
    #   So C[:, 0:2] = C_part1_0 + C_part1_1
    #   C[:, 2:4] = C_part2_0 + C_part2_1
    #   C[:, 4:6] = C_part3_0 + C_part3_1
    #   C[:, 6:7] = C_part4_0 + C_part4_1
    
    # Each C_part is computed independently. The total rank is the sum of ranks of all sub-problems.
    # 2 middle blocks * (3 Strassen(7) + 1 schoolbook(2)) = 2 * 23 = 46.
    
    # Let's verify: schoolbook rank is 2*4*7 = 56. We're reducing to 46.
    
    # Build the decomposition.
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Strassen's algorithm for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A @ B
    # 
    # Strassen uses 7 multiplications:
    # M1 = (a11+a22) * (b11+b22)
    # M2 = (a21+a22) * b11
    # M3 = a11 * (b12-b22)
    # M4 = a22 * (b21-b11)
    # M5 = (a11+a12) * b22
    # M6 = (a21-a11) * (b11+b12)
    # M7 = (a12-a22) * (b21+b22)
    # 
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M7
    
    # For each Strassen sub-problem, we need to create 7 rank-1 terms.
    # Each term has a coefficient in U, V, W.
    
    # Let's define a helper to add a Strassen decomposition for a specific sub-problem.
    # We'll track the current column index in U, V, W.
    
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 46
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 46
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 46
    
    col = 0
    
    # Strassen coefficients:
    # For each of the 7 multiplications, we have:
    # M1: U coeff for (a11+a22), V coeff for (b11+b22)
    # M2: U coeff for (a21+a22), V coeff for b11
    # M3: U coeff for a11, V coeff for (b12-b22)
    # M4: U coeff for a22, V coeff for (b21-b11)
    # M5: U coeff for (a11+a12), V coeff for b22
    # M6: U coeff for (a21-a11), V coeff for (b11+b12)
    # M7: U coeff for (a12-a22), V coeff for (b21+b22)
    
    # W coefficients for each output element:
    # C11 = M1 - M4 + M5 + M6 -> W[0,0]=1, W[0,1]=-1, W[0,2]=1, W[0,3]=1
    # C12 = M3 + M5 -> W[1,0]=1, W[1,1]=1
    # C21 = M2 + M4 -> W[2,0]=1, W[2,1]=1
    # C22 = M1 - M3 + M2 + M7 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,3]=1
    
    # Let's define the Strassen structure.
    # For a 2x2x2 sub-problem with A indices (i0,j0),(i0,j1),(i1,j0),(i1,j1)
    # and B indices (j0,k0),(j0,k1),(j1,k0),(j1,k1)
    # and C indices (k0,i0),(k0,i1),(k1,i0),(k1,i1)
    
    # We'll create a function to add a Strassen decomposition.
    
    def add_strassen(A_i0, A_i1, A_j0, A_j1, B_j0, B_j1, B_k0, B_k1, C_k0, C_k1, C_i0, C_i1):
        nonlocal col
        # A indices: (A_i0, A_j0), (A_i0, A_j1), (A_i1, A_j0), (A_i1, A_j1)
        # B indices: (B_j0, B_k0), (B_j0, B_k1), (B_j1, B_k0), (B_j1, B_k1)
        # C indices: (C_k0, C_i0), (C_k0, C_i1), (C_k1, C_i0), (C_k1, C_i1)
        
        # M1 = (a11+a22) * (b11+b22)
        # U: A_i0,A_j0 += 1; A_i1,A_j1 += 1
        # V: B_j0,B_k0 += 1; B_j1,B_k1 += 1
        U[A_i0 * m + A_j0][col] = 1.0
        U[A_i1 * m + A_j1][col] = 1.0
        V[B_j0 * p + B_k0][col] = 1.0
        V[B_j1 * p + B_k1][col] = 1.0
        # W: C11 += 1, C22 += 1
        W[C_k0 * n + C_i0][col] = 1.0
        W[C_k1 * n + C_i1][col] = 1.0
        col += 1
        
        # M2 = (a21+a22) * b11
        # U: A_i1,A_j0 += 1; A_i1,A_j1 += 1
        # V: B_j0,B_k0 += 1
        U[A_i1 * m + A_j0][col] = 1.0
        U[A_i1 * m + A_j1][col] = 1.0
        V[B_j0 * p + B_k0][col] = 1.0
        # W: C21 += 1, C22 += 1
        W[C_k0 * n + C_i1][col] = 1.0
        W[C_k1 * n + C_i1][col] = 1.0
        col += 1
        
        # M3 = a11 * (b12-b22)
        # U: A_i0,A_j0 += 1
        # V: B_j0,B_k1 += 1; B_j1,B_k1 -= 1
        U[A_i0 * m + A_j0][col] = 1.0
        V[B_j0 * p + B_k1][col] = 1.0
        V[B_j1 * p + B_k1][col] = -1.0
        # W: C12 += 1, C22 -= 1
        W[C_k0 * n + C_i1][col] = 1.0
        W[C_k1 * n + C_i1][col] = -1.0
        col += 1
        
        # M4 = a22 * (b21-b11)
        # U: A_i1,A_j1 += 1
        # V: B_j1,B_k0 += 1; B_j0,B_k0 -= 1
        U[A_i1 * m + A_j1][col] = 1.0
        V[B_j1 * p + B_k0][col] = 1.0
        V[B_j0 * p + B_k0][col] = -1.0
        # W: C11 -= 1, C21 += 1
        W[C_k0 * n + C_i0][col] = -1.0
        W[C_k0 * n + C_i1][col] = 1.0
        col += 1
        
        # M5 = (a11+a12) * b22
        # U: A_i0,A_j0 += 1; A_i0,A_j1 += 1
        # V: B_j1,B_k1 += 1
        U[A_i0 * m + A_j0][col] = 1.0
        U[A_i0 * m + A_j1][col] = 1.0
        V[B_j1 * p + B_k1][col] = 1.0
        # W: C11 += 1, C12 += 1
        W[C_k0 * n + C_i0][col] = 1.0
        W[C_k0 * n + C_i1][col] = 1.0
        col += 1
        
        # M6 = (a21-a11) * (b11+b12)
        # U: A_i1,A_j0 += 1; A_i0,A_j0 -= 1
        # V: B_j0,B_k0 += 1; B_j0,B_k1 += 1
        U[A_i1 * m + A_j0][col] = 1.0
        U[A_i0 * m + A_j0][col] = -1.0
        V[B_j0 * p + B_k0][col] = 1.0
        V[B_j0 * p + B_k1][col] = 1.0
        # W: C11 += 1
        W[C_k0 * n + C_i0][col] = 1.0
        col += 1
        
        # M7 = (a12-a22) * (b21+b22)
        # U: A_i0,A_j1 += 1; A_i1,A_j1 -= 1
        # V: B_j1,B_k0 += 1; B_j1,B_k1 += 1
        U[A_i0 * m + A_j1][col] = 1.0
        U[A_i1 * m + A_j1][col] = -1.0
        V[B_j1 * p + B_k0][col] = 1.0
        V[B_j1 * p + B_k1][col] = 1.0
        # W: C22 += 1
        W[C_k1 * n + C_i1][col] = 1.0
        col += 1
    
    def add_schoolbook_2x2x1(A_i0, A_i1, A_j0, A_j1, B_j0, B_j1, B_k0, C_k0, C_i0, C_i1):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        # Two rank-1 terms.
        
        # Term 1: A11 * B1 -> C1
        U[A_i0 * m + A_j0][col] = 1.0
        V[B_j0 * p + B_k0][col] = 1.0
        W[C_k0 * n + C_i0][col] = 1.0
        col += 1
        
        # Term 2: A12 * B2 -> C1
        U[A_i0 * m + A_j1][col] = 1.0
        V[B_j1 * p + B_k0][col] = 1.0
        W[C_k0 * n + C_i0][col] = 1.0
        col += 1
        
        # Term 3: A21 * B1 -> C2
        U[A_i1 * m + A_j0][col] = 1.0
        V[B_j0 * p + B_k0][col] = 1.0
        W[C_k0 * n + C_i1][col] = 1.0
        col += 1
        
        # Term 4: A22 * B2 -> C2
        U[A_i1 * m + A_j1][col] = 1.0
        V[B_j1 * p + B_k0][col] = 1.0
        W[C_k0 * n + C_i1][col] = 1.0
        col += 1
    
    # Now, iterate over middle blocks and p-subblocks.
    for b in range(2):  # Two middle blocks of size 2
        A_j0 = b * 2
        A_j1 = b * 2 + 1
        B_j0 = b * 2
        B_j1 = b * 2 + 1
        
        # p-subblocks: [0,1], [2,3], [4,5], [6]
        p_subblocks = [(0, 1), (2, 3), (4, 5), (6, 6)]
        
        for k0, k1 in p_subblocks:
            if k0 == k1:
                # 2x2x1 sub-problem
                # A: (0, A_j0), (0, A_j1), (1, A_j0), (1, A_j1)
                # B: (B_j0, k0), (B_j1, k0)
                # C: (k0, 0), (k0, 1)
                add_schoolbook_2x2x1(0, 1, A_j0, A_j1, B_j0, B_j1, k0, k0, 0, 1)
            else:
                # 2x2x2 sub-problem
                # A: (0, A_j0), (0, A_j1), (1, A_j0), (1, A_j1)
                # B: (B_j0, k0), (B_j0, k1), (B_j1, k0), (B_j1, k1)
                # C: (k0, 0), (k0, 1), (k1, 0), (k1, 1)
                add_strassen(0, 1, A_j0, A_j1, B_j0, B_j1, k0, k1, k0, k1, 0, 1)
    
    # Check if we finished within the deadline.
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # This shouldn't happen, but just in case.
        pass
    
    return {"U": U, "V": V, "W": W}
