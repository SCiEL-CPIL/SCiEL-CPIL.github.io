import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    
    # Part 1: 2x2x9 multiplication using Strassen (Rank 7)
    # A is 2x2 (cols 0,1 of original A)
    # B is 2x9 (rows 0,1 of original B)
    # Strassen decomposition for 2x2x2 extended to 2x2x9
    
    # U1, V1, W1 for the 7 terms
    # U is (n*m1, 7) -> (2*2, 7) = (4, 7)
    # V is (m1*p, 7) -> (2*9, 7) = (18, 7)
    # W is (p*n, 7) -> (9*2, 7) = (18, 7)
    
    U1 = np.zeros((n * m1, 7))
    V1 = np.zeros((m1 * p, 7))
    W1 = np.zeros((p * n, 7))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices for A (2x2):
    # A11 is (0,0) -> index 0
    # A12 is (0,1) -> index 1
    # A21 is (1,0) -> index 2
    # A22 is (1,1) -> index 3
    
    # Indices for B (2x9):
    # B11 is row 0, col 0..8 -> indices 0..8
    # B12 is row 1, col 0..8 -> indices 9..17
    
    # Indices for C (2x9) in W (p*n = 18):
    # W[k*n + i]
    # Row 0 of C: k=0..8, i=0 -> indices 0..8
    # Row 1 of C: k=0..8, i=1 -> indices 9..17
    
    # Term 1: M1
    # U1: A11 + A22. U1[0] += 1, U1[3] += 1. (U1 is 4x7, col 0)
    U1[0, 0] = 1
    U1[3, 0] = 1
    # V1: B11 + B22. V1[0..8] += 1, V1[9..17] += 1.
    V1[0:9, 0] = 1
    V1[9:18, 0] = 1
    # W1: C11 + C22. W1[0..8] += 1 (Row 0), W1[9..17] += 1 (Row 1)
    # Wait, M1 contributes to C11 and C22.
    # C11 is Row 0 of C. C22 is Row 1 of C.
    # So W1[0..8] += 1, W1[9..17] += 1.
    W1[0:9, 0] = 1
    W1[9:18, 0] = 1
    
    # Term 2: M2
    # U2: A21 + A22. U2[2] += 1, U2[3] += 1.
    U1[2, 1] = 1
    U1[3, 1] = 1
    # V2: B11. V2[0..8] += 1.
    V1[0:9, 1] = 1
    # W2: C21 - C22.
    # C21 is Row 1 of C? No.
    # Standard Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M4 + M7
    # C21 = M2 + M3
    # C22 = -M2 + M1 + M3 - M6
    # Here C11, C12, C21, C22 are blocks.
    # Since we didn't split k, C11 is Row 0, C21 is Row 1?
    # No.
    # A is 2x2. B is 2x9.
    # C = AB.
    # C[0, :] = A[0,0]B[0,:] + A[0,1]B[1,:]
    # C[1, :] = A[1,0]B[0,:] + A[1,1]B[1,:]
    # So C[0, :] corresponds to C11 (if B was 2x2, C11 is top-left).
    # C[1, :] corresponds to C21 (if B was 2x2, C21 is bottom-left).
    # So:
    # C11 -> Row 0 of C
    # C21 -> Row 1 of C
    
    # M2 contributes to C21 (Row 1) with +1 and C22 (Row 1? No C22 is bottom-right).
    # In 2x2x2, C22 is bottom-right.
    # Here, we have only one "block" of columns.
    # So C22 is not present?
    # Wait. The formula C22 = ... applies to the bottom-right block.
    # If we view B as 2x9, it's like 2x(2+2+2+2+1).
    # But we didn't split.
    # The Strassen algorithm computes C11, C12, C21, C22.
    # C11 and C21 are the left column of C (in 2x2 case).
    # C12 and C22 are the right column of C.
    # Here, the "left column" is the entire C (2x9).
    # So we only need C11 and C21?
    # No. The multiplication is $A_{2 \times 2} B_{2 \times 9}$.
    # This is equivalent to multiplying $A$ with 9 columns.
    # The Strassen formulas for C11 and C21 give the result for the first "block" of B.
    # If B is 2x9, we can treat it as 9 separate 2x1 multiplications?
    # No, Strassen mixes the rows of B.
    # The result C is 2x9.
    # The formulas for C11 and C21 give the 2x9 matrix C.
    # C11 in the formula corresponds to C[0, :].
    # C21 in the formula corresponds to C[1, :].
    # So we only care about C11 and C21?
    # Let's check.
    # C[0, :] = A[0,0]B[0,:] + A[0,1]B[1,:]
    # C[1, :] = A[1,0]B[0,:] + A[1,1]B[1,:]
    # Strassen C11 = M1 - M4 + M5 - M7
    # Strassen C21 = M2 + M3
    # Do these match?
    # Yes, if we define M's appropriately.
    # So W1 should only update the rows corresponding to C11 and C21.
    # C11 is Row 0 of C.
    # C21 is Row 1 of C.
    
    # Correcting W1 for M2:
    # M2 contributes to C21 (Row 1) with +1.
    # M2 contributes to C22 (Row 1? No C22 is not used).
    # So W1[9:18, 1] += 1.
    W1[9:18, 1] = 1
    
    # Term 3: M3
    # U3: A11. U3[0] += 1.
    U1[0, 2] = 1
    # V3: B21 - B22.
    # B21 is Row 1 of B? No.
    # In 2x2 B, B21 is bottom-left.
    # Here B is 2x9.
    # B21 corresponds to Row 1 of B?
    # Yes, B[1, :] is the bottom row.
    # So V3: B[1, :] - B[1, :]? No.
    # B21 is the element at (1,0) in 2x2.
    # In 2x9, B21 is the first element of the bottom row?
    # No. The Strassen decomposition assumes B is 2x2.
    # We are extending it to 2x9 by linearity.
    # B21 in the formula refers to the bottom-left block.
    # If we don't split columns, B21 is the entire bottom row?
    # No. B11 is top-left, B12 is top-right.
    # B21 is bottom-left, B22 is bottom-right.
    # If we view B as 2x9, we can't have left/right.
    # We must split the columns of B to use Strassen?
    # If we split B into 2x2 blocks, we need 4 or 5 blocks.
    # Rank would be 7 * 5 = 35.
    # But we found Rank 7 earlier.
    # Is Rank 7 valid for 2x2x9?
    # Yes, if we treat the 9 columns as independent.
    # The Strassen algorithm computes C from A and B.
    # The 7 intermediate products M1..M7 are linear combinations of A and B.
    # M1 = (A11+A22)(B11+B22).
    # If B is 2x9, B11+B22 is (Row 0 + Row 1).
    # So M1 is a vector.
    # The final C is reconstructed.
    # C11 = M1 - M4 + M5 - M7.
    # C21 = M2 + M3.
    # These give the two rows of C.
    # So we only need M1..M7.
    # And we only need to update W for C11 and C21.
    
    # V3: B21 - B22.
    # B21 is Row 1 of B? No.
    # In 2x2, B21 is bottom-left.
    # In 2x9, we don't have left/right.
    # However, the formula B21 - B22 implies a difference of two blocks.
    # If we don't split, B21 and B22 are not defined.
    # This implies we MUST split the columns of B to use Strassen directly?
    # If we split B into 2x2 blocks, rank is 7 * ceil(9/2) = 35.
    # If we don't split, can we use Strassen?
    # The Strassen algorithm is for 2x2 matrices.
    # For 2x9, we can use the 2x2 Strassen algorithm on the first 2 columns, then the next 2, etc.
    # This gives rank 7 * 5 = 35.
    # Can we do better?
    # The rank of 2x2xp is 7 * ceil(p/2) is an upper bound.
    # Is it the minimum?
    # For p=1, rank is 4.
    # For p=2, rank is 7.
    # For p=3, rank is 7+4=11?
    # So for p=9, rank is 7*4 + 4 = 32?
    # Or 7*5 = 35.
    
    # My previous Rank 7 argument was flawed because B21 and B22 are not defined for 2x9.
    # So Part 1 Rank is 35 (using 4 blocks of 2 and 1 block of 1).
    # 4 blocks of 2x2x2: Rank 7 * 4 = 28.
    # 1 block of 2x2x1: Rank 4.
    # Total 32.
    
    # Part 2: 2x1x9. Rank 1.
    # Total Rank 33.
    
    # This is better than 50.
    
    # Let's implement 32.
    
    # Split p=9 into 2,2,2,2,1.
    # U1, V1, W1 for each block.
    
    pass

# I will implement the 32 rank decomposition.
