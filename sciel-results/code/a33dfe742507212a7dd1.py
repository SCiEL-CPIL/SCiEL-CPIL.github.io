"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # We are multiplying a 2x4 matrix A by a 4x6 matrix B.
    # Partition A into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3).
    # Partition B into two 2x3 blocks: B_T (rows 0,1) and B_B (rows 2,3).
    # C = A_L @ B_T + A_R @ B_B.
    
    # B_T and B_B are 2x3. We can further partition them into 2x2 and 2x1 parts.
    # Let B_T = [B_T1 | B_T2] where B_T1 is 2x2 (cols 0,1) and B_T2 is 2x1 (col 2).
    # Let B_B = [B_B1 | B_B2] where B_B1 is 2x2 (cols 0,1) and B_B2 is 2x1 (col 2).
    
    # C is 2x6.
    # C[:, 0:2] = A_L @ B_T1 + A_R @ B_B1  (2x2 matrix)
    # C[:, 2:3] = A_L @ B_T2 + A_R @ B_B2  (2x1 vector)
    # C[:, 3:6] = ... wait, B has 6 columns.
    # B is 4x6. Rows 0,1 are B_T. Rows 2,3 are B_B.
    # B_T is 2x6. B_B is 2x6.
    # My previous partitioning was slightly off. Let's restart the partitioning carefully.
    
    # A (2x4) = [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3).
    # B (4x6) = [B1; B2] where B1 is 2x6 (rows 0,1), B2 is 2x6 (rows 2,3).
    # C (2x6) = A1 @ B1 + A2 @ B2.
    
    # Now we have two terms: A1 @ B1 and A2 @ B2.
    # A1, A2 are 2x2. B1, B2 are 2x6.
    # Let's decompose B1 and B2 into 2x2 blocks?
    # B1 (2x6) = [B11 | B12 | B13] where each is 2x2.
    # B2 (2x6) = [B21 | B22 | B23] where each is 2x2.
    
    # C = A1 @ [B11 | B12 | B13] + A2 @ [B21 | B22 | B23]
    # C = [ A1@B11 + A2@B21 | A1@B12 + A2@B22 | A1@B13 + A2@B23 ]
    
    # So C is composed of three 2x2 blocks: C1, C2, C3.
    # C1 = A1@B11 + A2@B21
    # C2 = A1@B12 + A2@B22
    # C3 = A1@B13 + A2@B23
    
    # Each Ci is a sum of two 2x2 matrix products.
    # We can compute A1@B11 using Strassen (rank 7).
    # We can compute A2@B21 using Strassen (rank 7).
    # But we need the SUM. Can we combine them?
    # Generally, no, unless we structure the inputs.
    # However, we are free to define the linear combinations in U, V, W.
    # The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    # The schoolbook rank is 48.
    # The rank of 2x2x2 is 7.
    # If we just use Strassen for each of the 6 individual 2x2 multiplications (A1@B11, A1@B12, A1@B13, A2@B21, A2@B22, A2@B23), we get 6 * 7 = 42.
    # Then we sum the results in W.
    # This is a valid rank-42 decomposition!
    
    # Let's verify this.
    # For each of the 6 blocks, we use 7 scalar multiplications.
    # Total 42 scalar multiplications.
    # The linearity of the reconstruction allows us to sum the contributions.
    
    # Let's construct the U, V, W matrices for this rank-42 solution.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U coefficients (for A part):
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1
    
    # V coefficients (for B part):
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B10=1, B11=-1
    # M4: B01=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B10=1
    # M7: B01=1, B11=1
    
    # W coefficients (for C part):
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: (none in C? Wait. M6 contributes to C10? No.
    # Let's re-verify Strassen.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # So:
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: C10=1? No. M6 is (A10-A00)(B00+B10).
    # Let's check the standard Strassen formulas again.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21-A11)(B11+B12)
    # M3 = (A11+A12)(B21-B22)
    # M4 = A11(B11-B21)
    # M5 = A22(B12-B11)
    # M6 = (A12+A22)(B21+B22)
    # M7 = (A11-A21)(B11+B12)
    # C11 = M1 - M5 + M4 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M2 + M7
    
    # There are different conventions. Let's use the one from the seed comment:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's verify C10 = M2 + M4.
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # M2+M4 = A10B00 + A11B01. Correct.
    
    # Let's verify C11 = M1 - M2 + M3 + M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # -M2 = -(A10+A11)B00 = -A10B00 - A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M7 = (A01-A11)(B01+B11) = A01B01 + A01B11 - A11B01 - A11B11
    # Sum:
    # A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A01B01 + A01B11 - A11B01 - A11B11
    # = A00B00 + A00B10 + A01B01 + A01B11 - A10B00 - A11B01
    # Wait. C11 should be A00B01 + A01B11? No.
    # C = A @ B.
    # C11 = A10B01 + A11B11.
    # My sum has A00B00... something is wrong.
    
    # Let's re-read the seed comment.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's check C11 again.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M7 = (A01-A11)(B01+B11)
    
    # Maybe the indices in the comment are 0-based but the formulas are for 1-based?
    # A00, A01, A10, A11.
    # C11 is the bottom-right element.
    # C11 = A10B01 + A11B11.
    
    # Let's use a known correct Strassen set.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21-A11)(B11+B12)
    # M3 = (A11+A12)(B21-B22)
    # M4 = A11(B11-B21)
    # M5 = A22(B12-B11)
    # M6 = (A12+A22)(B21+B22)
    # M7 = (A11-A21)(B11+B12)
    
    # C11 = M1 - M5 + M4 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M2 + M7
    
    # This is a standard set. Let's use this.
    # A is 2x2. A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1].
    # B is 2x2. B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1].
    # C is 2x2. C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1].
    
    # U coefficients (A part):
    # M1: A11=1, A22=1
    # M2: A21=1, A11=-1
    # M3: A11=1, A12=1
    # M4: A11=1
    # M5: A22=1
    # M6: A12=1, A22=1
    # M7: A11=1, A21=-1
    
    # V coefficients (B part):
    # M1: B11=1, B22=1
    # M2: B11=1, B12=1
    # M3: B21=1, B22=-1
    # M4: B11=1, B21=-1
    # M5: B12=1, B11=-1
    # M6: B21=1, B22=1
    # M7: B11=1, B12=1
    
    # W coefficients (C part):
    # M1: C11=1, C22=1
    # M2: C21=1, C22=-1
    # M3: C12=1, C22=-1
    # M4: C11=1, C21=-1
    # M5: C11=-1, C12=1
    # M6: C21=1
    # M7: C11=-1, C22=1
    
    # Let's verify C11 = M1 - M5 + M4 - M7.
    # M1: C11=1
    # -M5: C11=-(-1)=1
    # M4: C11=1
    # -M7: C11=-(-1)=1
    # Sum = 4? No.
    # C11 = M1 - M5 + M4 - M7.
    # Coeff of C11 in M1 is 1.
    # Coeff of C11 in M5 is -1. So -M5 gives +1.
    # Coeff of C11 in M4 is 1.
    # Coeff of C11 in M7 is -1. So -M7 gives +1.
    # Total 4. This is wrong.
    
    # I will not risk a broken Strassen implementation.
    # I will return the schoolbook solution which is guaranteed to be correct.
    # The seed program already does this.
    # Since I cannot find a reliable way to construct a lower rank solution in this time,
    # and the directive says "if I cannot find a valid lower rank, I should return the seed to get 1.0",
    # I will return the schoolbook solution.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
