"""Rewritten program using Strassen for 4x4x4 sub-problem and schoolbook for remainder."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build Strassen's 15-term decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns U, V, W matrices of shapes (4, 15), (4, 15), (4, 15)
    where U indexes input A (row, col), V indexes input B (row, col), 
    W indexes output C (row, col).
    
    Strassen's formulas:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B11 - B21)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: 4 rows (A11, A12, A21, A22), 15 cols
    U = np.zeros((4, 15))
    # V: 4 rows (B11, B12, B21, B22), 15 cols
    V = np.zeros((4, 15))
    # W: 4 rows (C11, C12, C21, C22), 15 cols
    W = np.zeros((4, 15))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 += M1, C22 += M1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 += M3, C22 += M3
    
    # M4 = A22(B11 - B21)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build rank-56 decomposition for 4x4 * 4x4 -> 4x4 using recursive Strassen.
    
    Returns U (16, 56), V (16, 56), W (16, 56)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # For 4x4, we split into 2x2 blocks and apply Strassen
    # Each of the 7 Strassen terms involves 2x2 multiplications
    # Each 2x2 multiplication has rank 7, so total rank = 7 * 7 = 49? 
    # Wait, no. Let me think again.
    
    # Actually, for 4x4 using Strassen:
    # We have 7 multiplications of 2x2 matrices
    # Each 2x2 multiplication can be done with rank 7 (Strassen)
    # So total rank = 7 * 7 = 49? No, that's not right either.
    
    # Let me reconsider. The 4x4 matrix multiplication tensor has rank 56 
    # when using Strassen's algorithm applied at the 4x4 level.
    
    # Actually, the standard result is that 4x4x4 has rank 56 using 
    # Strassen's algorithm recursively.
    
    # Let me build it properly. We'll use the fact that:
    # 4x4 * 4x4 = (2x2 blocks) * (2x2 blocks)
    # The 7 Strassen combinations give 7 sub-problems of 2x2 * 2x2
    # Each 2x2 * 2x2 has rank 7
    # So total rank = 7 * 7 = 49
    
    # Hmm, but the known rank of 4x4x4 is 56 with Strassen. Let me check.
    # Actually, I think the rank is 7^2 = 49 for the 4x4 case using 
    # Strassen recursively. But let me verify by construction.
    
    # Let me just build it directly.
    
    # For 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    
    # Strassen gives:
    # M1 = (A11 + A22)(B11 + B22)  -> 2x2 * 2x2, rank 7
    # M2 = (A21 + A22)B11           -> 2x2 * 2x2, rank 7
    # M3 = A11(B21 - B22)           -> 2x2 * 2x2, rank 7
    # M4 = A22(B11 - B21)           -> 2x2 * 2x2, rank 7
    # M5 = (A11 + A12)B22           -> 2x2 * 2x2, rank 7
    # M6 = (A21 - A11)(B11 + B12)   -> 2x2 * 2x2, rank 7
    # M7 = (A12 - A22)(B21 + B22)   -> 2x2 * 2x2, rank 7
    
    # Total rank = 7 * 7 = 49
    
    # But wait, the known minimal rank for 4x4x4 is not known to be 49.
    # Strassen gives an upper bound of 49 for 4x4x4.
    
    # Let me construct the 4x4x4 decomposition with rank 49.
    
    rank44 = 49
    U = np.zeros((16, rank44))
    V = np.zeros((16, rank44))
    W = np.zeros((16, rank44))
    
    # Index mapping for 2x2 blocks within 4x4:
    # Block (bi, bj) of A has rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # So A[2*bi + r, 2*bj + c] is element (r,c) of block (bi,bj)
    
    # For each of the 7 Strassen terms, and for each of the 7 sub-terms 
    # within each 2x2 multiplication, we add a rank-1 component.
    
    for term_idx in range(7):
        # Get the 2x2 Strassen factors for this term
        # U2[term_idx, :], V2[term_idx, :], W2[term_idx, :]
        # But we need to map 2x2 block indices to 4x4 indices
        
        # The 7 terms correspond to different block combinations
        # Let me define the block-level U, V, W for the 7 terms
        
        # Block-level (2x2 of 2x2 blocks):
        # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
        # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
        # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
        
        # For each term, the block-level coefficients are:
        # Term 0 (M1): A: [1,0,0,1], B: [1,0,0,1], C: [1,0,0,1]
        # Term 1 (M2): A: [0,0,1,1], B: [1,0,0,0], C: [0,0,1,-1]
        # Term 2 (M3): A: [1,0,0,0], B: [0,0,1,-1], C: [0,1,0,1]
        # Term 3 (M4): A: [0,0,0,1], B: [1,0,-1,0], C: [1,0,1,0]
        # Term 4 (M5): A: [1,1,0,0], B: [0,0,0,1], C: [-1,1,0,0]
        # Term 5 (M6): A: [0,0,1,-1], B: [1,1,0,0], C: [0,0,0,1]
        # Term 6 (M7): A: [0,1,0,-1], B: [0,0,1,1], C: [1,0,0,0]
        
        # But this is at the block level. Each "1" in the block-level 
        # coefficient means we need to multiply all 2x2 elements.
        
        # Actually, let me think of it differently. For each Strassen term,
        # we have a 2x2 * 2x2 multiplication. The 2x2 inputs are linear 
        # combinations of 2x2 blocks of the 4x4 matrices.
        
        # Let me define the block-level coefficients for each term:
        # A_block_coeffs[term][block_idx] where block_idx is 0..3 for 
        # (0,0), (0,1), (1,0), (1,1)
        
        A_block_coeffs = [
            [1, 0, 0, 1],  # M1: A11 + A22
            [0, 0, 1, 1],  # M2: A21 + A22
            [1, 0, 0, 0],  # M3: A11
            [0, 0, 0, 1],  # M4: A22
            [1, 1, 0, 0],  # M5: A11 + A12
            [0, 0, 1, -1], # M6: A21 - A11
            [0, 1, 0, -1], # M7: A12 - A22
        ]
        
        B_block_coeffs = [
            [1, 0, 0, 1],  # M1: B11 + B22
            [1, 0, 0, 0],  # M2: B11
            [0, 0, 1, -1], # M3: B21 - B22
            [1, 0, -1, 0], # M4: B11 - B21
            [0, 0, 0, 1],  # M5: B22
            [1, 1, 0, 0],  # M6: B11 + B12
            [0, 0, 1, 1],  # M7: B21 + B22
        ]
        
        C_block_coeffs = [
            [1, 0, 0, 1],  # M1: C11 + C22 (with + for both)
            [0, 0, 1, -1], # M2: C21 - C22 (with + for C21, - for C22)
            [0, 1, 0, 1],  # M3: C12 + C22
            [1, 0, 1, 0],  # M4: C11 + C21
            [-1, 1, 0, 0], # M5: -C11 + C12
            [0, 0, 0, 1],  # M6: C22
            [1, 0, 0, 0],  # M7: C11
        ]
        
        # For each term, we have a 2x2 * 2x2 multiplication with rank 7.
        # The 2x2 Strassen decomposition gives us U2, V2, W2 of shape (4, 7)
        # where the 4 rows correspond to the 4 elements of the 2x2 matrix.
        
        # For each of the 7 sub-terms within this 2x2 multiplication:
        for sub_idx in range(7):
            col = term_idx * 7 + sub_idx
            
            # For U: for each block of A with non-zero coefficient,
            # add the appropriate 2x2 element contributions
            for a_block in range(4):
                a_coeff = A_block_coeffs[term_idx][a_block]
                if a_coeff != 0:
                    # Block a_block has rows 2*(a_block//2) to 2*(a_block//2)+1
                    # and cols 2*(a_block%2) to 2*(a_block%2)+1
                    a_row_block = a_block // 2
                    a_col_block = a_block % 2
                    for r in range(2):
                        for c in range(2):
                            a_idx = (a_row_block * 2 + r) * 4 + (a_col_block * 2 + c)
                            # U2[sub_idx, r*2+c] is the coefficient for this element
                            U[a_idx, col] += a_coeff * U2[sub_idx, r * 2 + c]
            
            # For V: similar for B
            for b_block in range(4):
                b_coeff = B_block_coeffs[term_idx][b_block]
                if b_coeff != 0:
                    b_row_block = b_block // 2
                    b_col_block = b_block % 2
                    for r in range(2):
                        for c in range(2):
                            b_idx = (b_row_block * 2 + r) * 4 + (b_col_block * 2 + c)
                            V[b_idx, col] += b_coeff * V2[sub_idx, r * 2 + c]
            
            # For W: similar for C
            for c_block in range(4):
                c_coeff = C_block_coeffs[term_idx][c_block]
                if c_coeff != 0:
                    c_row_block = c_block // 2
                    c_col_block = c_block % 2
                    for r in range(2):
                        for c in range(2):
                            c_idx = (c_row_block * 2 + r) * 4 + (c_col_block * 2 + c)
                            W[c_idx, col] += c_coeff * W2[sub_idx, r * 2 + c]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4 * 4x5 -> 4x5.
    
    Uses Strassen for the 4x4 * 4x4 part (rank 49) and schoolbook for 
    the 4x4 * 4x1 part (rank 16), total rank 65.
    
    Wait, let me reconsider. The problem is 4x4 * 4x5 -> 4x5.
    We can split the 5 columns into 4 + 1.
    4x4 * 4x4 has rank 49 (using Strassen).
    4x4 * 4x1 has rank 16 (schoolbook).
    Total rank = 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Build 4x4x4 Strassen decomposition
    U44, V44, W44 = _build_strassen_4x4()
    rank44 = U44.shape[1]  # 49
    
    # Build 4x4x1 schoolbook decomposition
    rank41 = n * m * 1  # 16
    U41 = np.zeros((n * m, rank41))
    V41 = np.zeros((m * 1, rank41))
    W41 = np.zeros((1 * n, rank41))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(1):
                U41[i * m + j, col] = 1
                V41[j * 1 + k, col] = 1
                W41[k * n + i, col] = 1
                col += 1
    
    # Now combine:
    # U has shape (n*m, rank) = (16, 65)
    # V has shape (m*p, rank) = (20, 65)
    # W has shape (p*n, rank) = (20, 65)
    
    total_rank = rank44 + rank41  # 49 + 16 = 65
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Fill in the 4x4x4 part (first 49 columns)
    # U44 has shape (16, 49), V44 has shape (16, 49), W44 has shape (16, 49)
    # But V should have shape (20, 65) and W should have shape (20, 65)
    # The first 16 rows of V correspond to the 4x4 part (columns 0-3 of B)
    # The first 16 rows of W correspond to the 4x4 part (rows 0-3 of C)
    
    U[:, :rank44] = U44
    V[:m * 4, :rank44] = V44  # First 16 rows of V
    W[:4 * n, :rank44] = W44  # First 16 rows of W (4*4=16)
    
    # Fill in the 4x4x1 part (last 16 columns)
    # U41 has shape (16, 16), V41 has shape (4, 16), W41 has shape (4, 16)
    U[:, rank44:] = U41
    # V41 corresponds to column 4 of B (the 5th column)
    # B has 5 columns, so B[j, 4] is at row j*5 + 4 in V
    for j in range(m):
        V[j * p + 4, rank44:rank44 + rank41] = V41[j, :]
    # W41 corresponds to row 4 of C (the 5th row)
    # C has 5 rows, so C[4, i] is at row 4*n + i in W
    for i in range(n):
        W[4 * n + i, rank44:rank44 + rank41] = W41[i, :]
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
