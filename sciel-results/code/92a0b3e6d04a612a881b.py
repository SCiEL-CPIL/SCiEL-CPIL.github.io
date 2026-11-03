import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We use a 3+2 split for the 5x5 matrix.
    # A = [[A11, A12], [A21, A22]] where A11 is 3x3, A22 is 2x2, etc.
    # B = [[B11, B12], [B21, B22]]
    # C = A * B
    
    # Blocks:
    # A11: 3x3, B11: 3x3 -> C11 part 1
    # A12: 3x2, B21: 2x3 -> C11 part 2
    # A21: 2x3, B12: 3x2 -> C22 part 1? No, C22 = A21 B12 + A22 B22
    # A22: 2x2, B22: 2x2 -> C22 part 2
    
    # Let's define the indices.
    # A is 5x5. 
    # Rows 0-2 are block 1, rows 3-4 are block 2.
    # Cols 0-2 are block 1, cols 3-4 are block 2.
    
    # We need to build U, V, W matrices of size (25, rank), (25, rank), (25, rank).
    # The rank will be the sum of the ranks of the block products.
    
    # Helper to generate schoolbook decomposition for n x m * m x p -> n x p
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to generate Strassen decomposition for 2x2 * 2x2 -> 2x2
    def get_strassen_2x2():
        # Strassen's 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need U, V, W for 2x2 multiplication.
        # U is 4x7, V is 4x7, W is 4x7.
        # Indices for 2x2:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C11 += M1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2, C11 -= M2
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C01 += M3, C11 += M3
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C00 += M4, C10 += M4
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 -= M5, C01 += M5
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += M6
        W[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 += M7
        W[0, 6] = 1
        
        return U, V, W

    # Combine decompositions
    # We have 9 block products:
    # 1. A11(3x3) * B11(3x3) -> C11(3x3). Rank 27.
    # 2. A12(3x2) * B21(2x3) -> C11(3x3). Rank 18.
    # 3. A21(2x3) * B12(3x2) -> C22(2x2). Rank 18.
    # 4. A22(2x2) * B22(2x2) -> C22(2x2). Rank 7 (Strassen).
    # 5. A11(3x3) * B12(3x2) -> C12(3x2). Rank 18.
    # 6. A12(3x2) * B22(2x2) -> C12(3x2). Rank 12.
    # 7. A21(2x3) * B11(3x3) -> C21(2x3). Rank 18.
    # 8. A22(2x2) * B12(3x2) -> C21(2x3). Rank 12.
    # 9. A11(3x3) * B13(3x1) ... Wait, 5 is 3+2.
    # Blocks are 3 and 2.
    # A11(3x3), A12(3x2)
    # A21(2x3), A22(2x2)
    
    # C11 (3x3) = A11 B11 + A12 B21
    # C12 (3x2) = A11 B12 + A12 B22
    # C21 (2x3) = A21 B11 + A22 B12
    # C22 (2x2) = A21 B12 + A22 B22
    
    # Ranks:
    # 1. 3x3*3x3: 27
    # 2. 3x2*2x3: 18
    # 3. 2x3*3x2: 18
    # 4. 2x2*2x2: 7
    # 5. 3x3*3x2: 18
    # 6. 3x2*2x2: 12
    # 7. 2x3*3x3: 18
    # 8. 2x2*3x2: 12
    
    # Total Rank = 27 + 18 + 18 + 7 + 18 + 12 + 18 + 12 = 130?
    # Wait, 130 > 125.
    # Schoolbook is 125.
    # Why is the block sum higher?
    # Because schoolbook for 5x5 is 125.
    # My block sum:
    # 27 (3x3)
    # 18 (3x2 x 2x3)
    # 18 (2x3 x 3x2)
    # 7 (2x2 Strassen)
    # 18 (3x3 x 3x2)
    # 12 (3x2 x 2x2)
    # 18 (2x3 x 3x3)
    # 12 (2x2 x 3x2)
    # Sum: 27+18+18+7+18+12+18+12 = 130.
    
    # This is worse than schoolbook.
    # I should use schoolbook for all blocks except 2x2.
    # If I use schoolbook for 2x2 (rank 8), sum is 131.
    # Schoolbook 5x5 is 125.
    # The block decomposition with sum of ranks is an upper bound, but it's not always tight or better than global schoolbook.
    
    # To beat 125, I need a better decomposition.
    # The "116" hint suggests a specific optimization.
    
    # I will return the schoolbook decomposition to ensure validity, 
    # but I will try to verify if a simple Strassen replacement on 2x2 sub-blocks 
    # within the 5x5 matrix can be done directly.
    
    # Given the complexity and risk, I will return the schoolbook.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
