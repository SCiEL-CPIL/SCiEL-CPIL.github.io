import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 4x4x4 (first 4 columns of B)
    # Rank 49
    U_strassen, V_strassen, W_strassen = _strassen_4x4x4()
    
    # 2. Schoolbook for 4x4x1 (5th column of B)
    # Rank 4
    U_col, V_col, W_col = _schoolbook_4x4x1()
    
    # Combine
    U = [row + [0]*4 for row in U_strassen]
    V = [row + [0]*4 for row in V_strassen]
    W = [row + [0]*4 for row in W_strassen]
    
    for i in range(4):
        U[i*4 + 0].append(U_col[i]) # Wait, U_strassen is 16x49. U_col is 4x4.
        # We need to append the U_col entries to the corresponding rows of U.
        # U_strassen has 16 rows (i,j).
        # U_col has 4 rows (i).
        # The 5th column calculation depends on A[i,j] for all j.
        # So for each i, we add 4 terms.
        # The rows of U are indexed by i*m + j.
        # For the 5th column, U[i*m + j] should have 1 in the new columns for the j-th term of the i-th output.
        
    # Let's do it properly.
    rank_strassen = len(U_strassen[0])
    rank_col = len(U_col[0])
    total_rank = rank_strassen + rank_col
    
    U_final = [[0] * total_rank for _ in range(n * m)]
    V_final = [[0] * total_rank for _ in range(m * p)]
    W_final = [[0] * total_rank for _ in range(p * n)]
    
    # Copy Strassen
    for r in range(rank_strassen):
        for i in range(n * m):
            U_final[i][r] = U_strassen[i][r]
        for j in range(m * 4): # m=4, p=4 for Strassen part
            V_final[j][r] = V_strassen[j][r]
        for k in range(4 * n): # p=4, n=4
            W_final[k][r] = W_strassen[k][r]
            
    # Add Column part
    # The column part is A (4x4) * B_col (4x1) = C_col (4x1)
    # C_col[i] = sum_j A[i,j] B_col[j]
    # Terms: for each i, j: U[i*4+j, r] = 1, V[j*5+4, r] = 1, W[4*4+i, r] = 1
    # Note: W index is k*n + i. Here k=4 (5th column).
    # So W index is 4*4 + i = 16 + i.
    
    for i in range(4):
        for j in range(4):
            r = rank_strassen + i * 4 + j
            U_final[i * 4 + j][r] = 1
            V_final[j * 5 + 4][r] = 1 # j is row of B, 4 is col index
            W_final[4 * 4 + i][r] = 1 # k=4, i is row of C
            
    return {"U": U_final, "V": V_final, "W": W_final}

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 multiplication using Strassen.
    U: 16x49
    V: 16x49
    W: 16x49
    """
    # First, get 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Strassen formulas for 4x4:
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We map the 7 M's to 7 sets of 7 rank-1 terms.
    # Total 49 terms.
    
    # Helper to add a 2x2x2 decomposition to the 4x4x4 matrices
    # with specific coefficients for A, B, C blocks.
    
    # Indices for 2x2 blocks in 4x4:
    # A block (I, J) corresponds to rows I*2..I*2+1, cols J*2..J*2+1
    # U index: i*4 + j.
    # If we want to multiply A_block(I1, J1) by B_block(I2, J2),
    # we use the 2x2x2 decomposition, scaling U by coeff_A, V by coeff_B.
    
    # Let's define the 7 M terms.
    # Each M_t is a product of two 2x2 matrices.
    # We need to express A^{(t)} and B^{(t)} as linear combinations of the 4 blocks of A and B.
    
    # M1: A1 = A11 + A22, B1 = B11 + B22
    # M2: A2 = A21 + A22, B2 = B11
    # M3: A3 = A11, B3 = B12 - B22
    # M4: A4 = A22, B4 = B21 - B11
    # M5: A5 = A11 + A12, B5 = B22
    # M6: A6 = A21 - A11, B6 = B11 + B12
    # M7: A7 = A12 - A22, B7 = B21 + B22
    
    # C blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate through the 7 M's.
    # For each M_t, we have a set of 7 rank-1 terms from the 2x2x2 decomposition.
    # We add them to U, V, W with appropriate coefficients.
    
    # U2, V2, W2 are 4x7.
    # U2[i,j] is the coefficient of A[i,j] in the i-th row of U for the j-th term.
    
    # For M_t, A^{(t)} is a 2x2 matrix.
    # A^{(t)}[i,j] = sum_{I,J} coeff_A_t[i,j,I,J] * A[I*2+i, J*2+j]
    # Similarly for B.
    
    # We define the coefficients for each M_t.
    # A_t is 2x2. B_t is 2x2.
    # We need to map the 2x2 indices (i,j) to 4x4 indices (I*2+i, J*2+j).
    
    # Let's create a function to add a term.
    def add_term(m_idx, coeff_A, coeff_B, coeff_C):
        # m_idx: 0-6
        # coeff_A: 2x2 matrix of coefficients for A blocks
        # coeff_B: 2x2 matrix of coefficients for B blocks
        # coeff_C: 2x2 matrix of coefficients for C blocks (output)
        
        # coeff_A[i,j] means the (i,j) entry of A^{(t)} is sum coeff_A[i,j,I,J] * A[I,J]
        # Wait, A^{(t)} is a 2x2 matrix.
        # A^{(t)}[i,j] is formed from A blocks.
        # Actually, A^{(t)} is a linear combination of the 4 A blocks.
        # A^{(t)} = sum_{I,J} w_A[I,J] * A[I,J]
        # Where w_A is a 2x2 matrix of 2x2 matrices? No.
        # A is 4x4. A[I,J] is 2x2.
        # A^{(t)} is 2x2.
        # So A^{(t)}[i,j] = sum_{I,J} w_A_t[I,J] * A[I*2+i, J*2+j]
        
        # We have U2, V2, W2 for 2x2x2.
        # U2[i,j, r] is coeff of A[i,j] in term r.
        
        for r in range(7):
            global_r = m_idx * 7 + r
            
            # Update U
            for i in range(2):
                for j in range(2):
                    # U2[i,j,r] is the coefficient for A[i,j] in this term
                    u_val = U2[i*2+j][r]
                    if u_val != 0:
                        for I in range(2):
                            for J in range(2):
                                # A^{(t)}[i,j] gets contribution from A[I,J]
                                # The coefficient is w_A_t[I,J]
                                # Wait, w_A_t is a scalar?
                                # No, A^{(t)} is a 2x2 matrix.
                                # A^{(t)}[i,j] = sum_{I,J} w_A_t[I,J] * A[I*2+i, J*2+j]
                                # So for each i,j, we add u_val * w_A_t[I,J] to U[I*2+i, J*2+j, global_r]
                                w_val = coeff_A[I*2+i, J*2+j] # This is not right.
                                # coeff_A is 2x2. It tells us how A blocks are combined.
                                # A^{(t)}[i,j] = sum_{I,J} coeff_A[I,J] * A[I,J][i,j]
                                # So the coefficient for A[I*2+i, J*2+j] is coeff_A[I,J].
                                w_val = coeff_A[I,J]
                                if w_val != 0:
                                    U[I*2+i, J*2+j][global_r] += u_val * w_val
                                    
            # Update V
            for i in range(2):
                for j in range(2):
                    v_val = V2[i*2+j][r]
                    if v_val != 0:
                        for I in range(2):
                            for J in range(2):
                                w_val = coeff_B[I,J]
                                if w_val != 0:
                                    V[I*2+i, J*2+j][global_r] += v_val * w_val
                                    
            # Update W
            for i in range(2):
                for j in range(2):
                    w_val_out = W2[i*2+j][r]
                    if w_val_out != 0:
                        for I in range(2):
                            for J in range(2):
                                # C[I,J][i,j] gets contribution from M_t[i,j]
                                # C[I,J] = sum_{K,L} coeff_C[I,J] * M_t[K,L]
                                # M_t[K,L] is the (K,L) entry of the 2x2 product.
                                # W2[i,j,r] is coeff of C[i,j] in term r.
                                # So for each output C[I*2+i, J*2+j], we add w_val_out * coeff_C[I,J]
                                w_coeff = coeff_C[I,J]
                                if w_coeff != 0:
                                    W[I*2+i, J*2+j][global_r] += w_val_out * w_coeff

    # Define coeff_A, coeff_B, coeff_C for each M
    # M1: A = A11+A22, B = B11+B22
    # coeff_A[0,0]=1, [1,1]=1, others 0
    # coeff_B[0,0]=1, [1,1]=1, others 0
    # C11 gets +1, C22 gets +1
    
    # We need to call add_term for each M and each C block it contributes to.
    # Or, we can sum the contributions.
    
    # Let's define the contributions clearly.
    # M1 contributes to C11 (+1) and C22 (+1)
    # M2 contributes to C21 (+1) and C22 (-1)
    # M3 contributes to C12 (+1) and C22 (+1)
    # M4 contributes to C11 (+1) and C21 (+1)
    # M5 contributes to C11 (-1) and C12 (+1)
    # M6 contributes to C22 (+1)
    # M7 contributes to C11 (+1)
    
    # We will iterate m from 0 to 6.
    # For each m, we determine coeff_A, coeff_B.
    # Then we determine which C blocks it contributes to and with what sign.
    
    configs = [
        # (coeff_A, coeff_B, [(cI, cJ, sign), ...])
        ([[1,0],[0,1]], [[1,0],[0,1]], [(0,0,1), (1,1,1)]), # M1
        ([[0,0],[1,1]], [[1,0],[0,0]], [(1,0,1), (1,1,-1)]), # M2
        ([[1,0],[0,0]], [[0,1],[0,-1]], [(0,1,1), (1,1,1)]), # M3
        ([[0,0],[0,1]], [[0,-1],[1,0]], [(0,0,1), (1,0,1)]), # M4
        ([[1,1],[0,0]], [[0,0],[0,1]], [(0,0,-1), (0,1,1)]), # M5
        ([[0,-1],[1,0]], [[1,1],[0,0]], [(1,1,1)]), # M6
        ([[0,1],[0,-1]], [[0,1],[1,1]], [(0,0,1)]) # M7
    ]
    
    for m_idx, (cA, cB, c_contribs) in enumerate(configs):
        # cA, cB are 2x2 lists
        # c_contribs is list of (I, J, sign)
        
        # We need to add this M to the decomposition.
        # But add_term expects coeff_C to be a 2x2 matrix.
        # We can call add_term 7 times? No, add_term adds one M's worth of 7 terms.
        # We need to sum the W contributions.
        
        # Let's modify add_term to accept a list of contributions.
        # Or just loop over contributions and add W separately.
        
        # U and V are the same for all contributions of M_t.
        # W depends on the contribution.
        
        # So we compute U and V parts once for M_t.
        # Then for each contribution, we add the W part.
        
        # U and V calculation
        for r in range(7):
            global_r = m_idx * 7 + r
            
            for i in range(2):
                for j in range(2):
                    u_val = U2[i*2+j][r]
                    if u_val != 0:
                        for I in range(2):
                            for J in range(2):
                                w_val = cA[I,J]
                                if w_val != 0:
                                    U[I*2+i, J*2+j][global_r] += u_val * w_val
                                    
            for i in range(2):
                for j in range(2):
                    v_val = V2[i*2+j][r]
                    if v_val != 0:
                        for I in range(2):
                            for J in range(2):
                                w_val = cB[I,J]
                                if w_val != 0:
                                    V[I*2+i, J*2+j][global_r] += v_val * w_val
                                    
            # W calculation
            for cI, cJ, sign in c_contribs:
                for i in range(2):
                    for j in range(2):
                        w_val_out = W2[i*2+j][r]
                        if w_val_out != 0:
                            W[cI*2+i, cJ*2+j][global_r] += w_val_out * sign

    return U, V, W

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen.
    U: 4x7
    V: 4x7
    W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A: [1,0,0,1] -> U[0]+=1, U[3]+=1
    # B: [1,0,0,1] -> V[0]+=1, V[3]+=1
    # C: C11 += M1, C22 += M1
    # W: [1,0,0,1] -> W[0]+=1, W[3]+=1
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    # A: [0,0,1,1]
    # B: [1,0,0,0]
    # C: C21 += M2, C22 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12-B22)
    # A: [1,0,0,0]
    # B: [0,1,0,-1]
    # C: C12 += M3, C22 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    # A: [0,0,0,1]
    # B: [0,-1,1,0]
    # C: C11 += M4, C21 += M4
    U[3][3] = 1
    V[1][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    # A: [1,1,0,0]
    # B: [0,0,0,1]
    # C: C11 -= M5, C12 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A: [-1,0,1,0]
    # B: [1,1,0,0]
    # C: C22 += M6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A: [0,1,0,-1]
    # B: [0,0,1,1]
    # C: C11 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_4x4x1():
    """
    Returns U, V, W for 4x4x1 multiplication.
    U: 4x4
    V: 4x4
    W: 4x4
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            # Wait, rank is 16?
            # No, I said rank 4 earlier.
            # C[i] = sum_j A[i,j] B[j]
            # This is 4 dot products.
            # Each dot product has rank 4? No.
            # The tensor is sum_i,j,k A[i,j] B[j] C[i]
            # This is not rank 1.
            # The schoolbook rank is 4*4*1 = 16.
            
    # Let's return rank 16.
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(16)]
    W = [[0]*16 for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i*4+j][r] = 1
            V[j][r] = 1 # V index is j (row of B). B is 4x1.
            W[i][r] = 1 # W index is i (row of C). C is 4x1.
            
    return U, V, W

