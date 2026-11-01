import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=4
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (8, rank)
    # W: (p*n, rank) = (8, rank)
    #
    # We want to compute C = A * B where A is 2x2 and B is 2x4.
    # C is 2x4.
    #
    # Known result: mu(2,2,4) = 12.
    # Let's try to construct a rank 12 solution.
    #
    # A = [[a00, a01],
    #      [a10, a11]]
    #
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13]]
    #
    # C = A * B
    # C = [[a00b00+a10b00, a00b01+a10b01, a00b02+a10b02, a00b03+a10b03],
    #      [a00b10+a10b10, a00b11+a10b11, a00b12+a10b12, a00b13+a10b13]]
    #
    # Wait, A is 2x2, B is 2x4.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # C[0,0] = a00 b00 + a01 b10
    # C[0,1] = a00 b01 + a01 b11
    # C[0,2] = a00 b02 + a01 b12
    # C[0,3] = a00 b03 + a01 b13
    # C[1,0] = a10 b00 + a11 b10
    # C[1,1] = a10 b01 + a11 b11
    # C[1,2] = a10 b02 + a11 b12
    # C[1,3] = a10 b03 + a11 b13
    #
    # Let's use Strassen-like ideas.
    # We can group the columns of B.
    # Let B1 = B[:, 0:2] and B2 = B[:, 2:4].
    # Then C1 = A * B1 and C2 = A * B2.
    # Each is a 2x2 * 2x2 multiplication.
    # Rank of 2x2 * 2x2 is 7 (Strassen).
    # So rank of 2x2 * 2x4 is at most 7 + 7 = 14.
    #
    # Can we do better?
    # Yes, because A is the same in both.
    #
    # Let's look for a rank 12 solution.
    #
    # Consider the following 12 products:
    #
    # Let's define the U, V, W matrices.
    #
    # A known rank 12 decomposition for 2x2 * 2x4:
    #
    # M1 = (a00+a11)(b00+b10)
    # M2 = (a00+a11)(b01+b11)
    # M3 = (a00+a11)(b02+b12)
    # M4 = (a00+a11)(b03+b13)
    # M5 = (a00-a11)(b00-b10)
    # M6 = (a00-a11)(b01-b11)
    # M7 = (a00-a11)(b02-b12)
    # M8 = (a00-a11)(b03-b13)
    # M9 = a00(b00+b01+b02+b03)  -- No, this is not bilinear in the right way.
    #
    # Let's try to derive it.
    #
    # C[0,j] = a00 b0j + a01 b1j
    # C[1,j] = a10 b0j + a11 b1j
    #
    # Let's use the following 12 terms:
    #
    # 1. (a00+a10)(b00+b01)
    # 2. (a00+a10)(b02+b03)
    # 3. (a01+a11)(b10+b11)
    # 4. (a01+a11)(b12+b13)
    # 5. (a00-a10)(b00-b01)
    # 6. (a00-a10)(b02-b03)
    # 7. (a01-a11)(b10-b11)
    # 8. (a01-a11)(b12-b13)
    # 9. (a00+a01)(b00+b10)
    # 10. (a00+a01)(b01+b11)
    # 11. (a10+a11)(b02+b12)
    # 12. (a10+a11)(b03+b13)
    #
    # This is getting complicated.
    #
    # Let's stick to the safe Rank 14 solution which is known to be valid.
    # The score is 1.1429.
    #
    # I will implement the Rank 14 solution carefully.
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For B1 (columns 0,1):
    # M1_1 = (a00+a11)(b00+b11)
    # M2_1 = (a10+a11)b00
    # M3_1 = a00(b01-b11)
    # M4_1 = a11(b10-b00)
    # M5_1 = (a00+a01)b11
    # M6_1 = (a10-a00)(b00+b01)
    # M7_1 = (a01-a11)(b10+b11)
    #
    # C[0,0] = M1_1 - M4_1 + M5_1 - M7_1
    # C[0,1] = M3_1 + M5_1
    # C[1,0] = M2_1 + M4_1
    # C[1,1] = M1_1 - M2_1 + M3_1 + M6_1
    #
    # For B2 (columns 2,3):
    # M1_2 = (a00+a11)(b02+b13)
    # M2_2 = (a10+a11)b02
    # M3_2 = a00(b03-b13)
    # M4_2 = a11(b12-b02)
    # M5_2 = (a00+a01)b13
    # M6_2 = (a10-a00)(b02+b03)
    # M7_2 = (a01-a11)(b12+b13)
    #
    # C[0,2] = M1_2 - M4_2 + M5_2 - M7_2
    # C[0,3] = M3_2 + M5_2
    # C[1,2] = M2_2 + M4_2
    # C[1,3] = M1_2 - M2_2 + M3_2 + M6_2
    #
    # Total rank = 7 + 7 = 14.
    
    rank = 14
    
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # Helper to add a Strassen block
    def add_strassen_block(offset, col_offset):
        # col_offset is 0 for B1, 2 for B2
        # B indices:
        # B1: b00, b01, b10, b11 -> V indices: 0, 1, 4, 5
        # B2: b02, b03, b12, b13 -> V indices: 2, 3, 6, 7
        
        # M1: (a00+a11)(b00+b11)
        # U: a00 (0), a11 (3)
        # V: b00 (0), b11 (5) for B1; b02 (2), b13 (7) for B2
        # W: C00 (0), C11 (3) for B1; C02 (2), C13 (7) for B2
        
        # r = offset
        U[0, offset] = 1
        U[3, offset] = 1
        V[col_offset, offset] = 1
        V[col_offset + 5, offset] = 1
        W[col_offset, offset] = 1
        W[col_offset + 3, offset] = 1
        
        # M2: (a10+a11)b00
        # U: a10 (2), a11 (3)
        # V: b00 (0) for B1; b02 (2) for B2
        # W: C10 (1), C11 (3) for B1; C12 (5), C13 (7) for B2
        
        # r = offset + 1
        U[2, offset + 1] = 1
        U[3, offset + 1] = 1
        V[col_offset, offset + 1] = 1
        W[col_offset + 1, offset + 1] = 1
        W[col_offset + 3, offset + 1] = -1
        
        # M3: a00(b01-b11)
        # U: a00 (0)
        # V: b01 (1), b11 (5) for B1; b03 (3), b13 (7) for B2
        # W: C01 (2), C11 (3) for B1; C03 (6), C13 (7) for B2
        
        # r = offset + 2
        U[0, offset + 2] = 1
        V[col_offset + 1, offset + 2] = 1
        V[col_offset + 5, offset + 2] = -1
        W[col_offset + 2, offset + 2] = 1
        W[col_offset + 3, offset + 2] = 1
        
        # M4: a11(b10-b00)
        # U: a11 (3)
        # V: b10 (4), b00 (0) for B1; b12 (6), b02 (2) for B2
        # W: C00 (0), C10 (1) for B1; C02 (2), C12 (5) for B2
        
        # r = offset + 3
        U[3, offset + 3] = 1
        V[col_offset + 4, offset + 3] = 1
        V[col_offset, offset + 3] = -1
        W[col_offset, offset + 3] = -1
        W[col_offset + 1, offset + 3] = 1
        
        # M5: (a00+a01)b11
        # U: a00 (0), a01 (1)
        # V: b11 (5) for B1; b13 (7) for B2
        # W: C00 (0), C01 (2) for B1; C02 (2), C03 (6) for B2
        
        # r = offset + 4
        U[0, offset + 4] = 1
        U[1, offset + 4] = 1
        V[col_offset + 5, offset + 4] = 1
        W[col_offset, offset + 4] = -1
        W[col_offset + 2, offset + 4] = 1
        
        # M6: (a10-a00)(b00+b01)
        # U: a10 (2), a00 (0)
        # V: b00 (0), b01 (1) for B1; b02 (2), b03 (3) for B2
        # W: C11 (3) for B1; C13 (7) for B2
        
        # r = offset + 5
        U[2, offset + 5] = 1
        U[0, offset + 5] = -1
        V[col_offset, offset + 5] = 1
        V[col_offset + 1, offset + 5] = 1
        W[col_offset + 3, offset + 5] = 1
        
        # M7: (a01-a11)(b10+b11)
        # U: a01 (1), a11 (3)
        # V: b10 (4), b11 (5) for B1; b12 (6), b13 (7) for B2
        # W: C00 (0) for B1; C02 (2) for B2
        
        # r = offset + 6
        U[1, offset + 6] = 1
        U[3, offset + 6] = -1
        V[col_offset + 4, offset + 6] = 1
        V[col_offset + 5, offset + 6] = 1
        W[col_offset, offset + 6] = -1
        
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    
    return {"U": U, "V": V, "W": W}
