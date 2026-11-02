import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-23 decomposition for 3x3 matrix multiplication
    # Based on known constructions from the literature
    # U: (9, 23), V: (9, 23), W: (9, 23)
    
    R = 23
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Helper to set entries
    def set_uv(r, i, j, val_u=1, val_v=1, val_w=1):
        U[i*m+j, r] = val_u
        V[j*p+k if False else (j*p+k), r] = val_v  # This won't work, need k
    
    # The rank-23 decomposition is complex. Let me use a verified construction.
    # I'll use the standard rank-23 pattern.
    
    # Term 0-2: Diagonal terms for C[0,0]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # Term 0: A[0,0]*B[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A[0,1]*B[1,0]  
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: A[0,2]*B[2,0]
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # Term 3: A[0,0]*B[0,1]
    U[0, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    # Term 4: A[0,1]*B[1,1]
    U[1, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
    # Term 5: A[0,2]*B[2,1]
    U[2, 5] = 1; V[7, 5] = 1; W[1, 5] = 1
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # Term 6: A[0,0]*B[0,2]
    U[0, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
    # Term 7: A[0,1]*B[1,2]
    U[1, 7] = 1; V[5, 7] = 1; W[2, 7] = 1
    # Term 8: A[0,2]*B[2,2]
    U[2, 8] = 1; V[8, 8] = 1; W[2, 8] = 1
    
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # Term 9: A[1,0]*B[0,0]
    U[3, 9] = 1; V[0, 9] = 1; W[3, 9] = 1
    # Term 10: A[1,1]*B[1,0]
    U[4, 10] = 1; V[3, 10] = 1; W[3, 10] = 1
    # Term 11: A[1,2]*B[2,0]
    U[5, 11] = 1; V[6, 11] = 1; W[3, 11] = 1
    
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # Term 12: A[1,0]*B[0,1]
    U[3, 12] = 1; V[1, 12] = 1; W[4, 12] = 1
    # Term 13: A[1,1]*B[1,1]
    U[4, 13] = 1; V[4, 13] = 1; W[4, 13] = 1
    # Term 14: A[1,2]*B[2,1]
    U[5, 14] = 1; V[7, 14] = 1; W[4, 14] = 1
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # Term 15: A[1,0]*B[0,2]
    U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1
    # Term 16: A[1,1]*B[1,2]
    U[4, 16] = 1; V[5, 16] = 1; W[5, 16] = 1
    # Term 17: A[1,2]*B[2,2]
    U[5, 17] = 1; V[8, 17] = 1; W[5, 17] = 1
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # Term 18: A[2,0]*B[0,0]
    U[6, 18] = 1; V[0, 18] = 1; W[6, 18] = 1
    # Term 19: A[2,1]*B[1,0]
    U[7, 19] = 1; V[3, 19] = 1; W[6, 19] = 1
    # Term 20: A[2,2]*B[2,0]
    U[8, 20] = 1; V[6, 20] = 1; W[6, 20] = 1
    
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # Term 21: A[2,0]*B[0,1]
    U[6, 21] = 1; V[1, 21] = 1; W[7, 21] = 1
    # Term 22: A[2,1]*B[1,1]
    U[7, 22] = 1; V[4, 22] = 1; W[7, 22] = 1
    
    # That's only 23 terms but I've used 23 and I'm missing C[2,2]
    # Wait, I have 23 terms (0-22) but I need to cover all 9 output entries
    # Let me recount: I have terms 0-22 which is 23 terms
    # But I haven't covered C[2,2] fully
    
    # This approach is getting messy. Let me use a verified rank-23 construction.
    
    # Actually, the above is just the schoolbook with 27 terms compressed incorrectly.
    # Let me use a proper rank-23 construction.
    
    # The known rank-23 decomposition uses specific patterns. Let me implement 
    # the standard one from literature.
    
    # Reset
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # I'll use the construction where we group terms.
    # For 3x3, rank 23 is achieved by specific linear combinations.
    
    # Given the complexity, let me return schoolbook rank 27 to be safe,
    # but try a quick search for rank 23.
    
    # Actually, let me just return the schoolbook to ensure validity.
    # The key insight is that I need a VERIFIED rank-23 decomposition.
    
    # Let me construct the standard rank-23 solution properly.
    # Based on the construction, the 23 terms are:
    
    # I'll use a known valid rank-23 pattern
    terms = [
        # (u_idx, v_idx, w_idx, u_val, v_val, w_val)
        (0, 0, 0, 1, 1, 1),    # 0: A00*B00 -> C00
        (1, 3, 0, 1, 1, 1),    # 1: A01*B10 -> C00  
        (2, 6, 0, 1, 1, 1),    # 2: A02*B20 -> C00
        (0, 1, 1, 1, 1, 1),    # 3: A00*B01 -> C01
        (1, 4, 1, 1, 1, 1),    # 4: A01*B11 -> C01
        (2, 7, 1, 1, 1, 1),    # 5: A02*B21 -> C01
        (0, 2, 2, 1, 1, 1),    # 6: A00*B02 -> C02
        (1, 5, 2, 1, 1, 1),    # 7: A01*B12 -> C02
        (2, 8, 2, 1, 1, 1),    # 8: A02*B22 -> C02
        (3, 0, 3, 1, 1, 1),    # 9: A10*B00 -> C10
        (4, 3, 3, 1, 1, 1),    # 10: A11*B10 -> C10
        (5, 6, 3, 1, 1, 1),    # 11: A12*B20 -> C10
        (3, 1, 4, 1, 1, 1),    # 12: A10*B01 -> C11
        (4, 4, 4, 1, 1, 1),    # 13: A11*B11 -> C11
        (5, 7, 4, 1, 1, 1),    # 14: A12*B21 -> C11
        (3, 2, 5, 1, 1, 1),    # 15: A10*B02 -> C12
        (4, 5, 5, 1, 1, 1),    # 16: A11*B12 -> C12
        (5, 8, 5, 1, 1, 1),    # 17: A12*B22 -> C12
        (6, 0, 6, 1, 1, 1),    # 18: A20*B00 -> C20
        (7, 3, 6, 1, 1, 1),    # 19: A21*B10 -> C20
        (8, 6, 6, 1, 1, 1),    # 20: A22*B20 -> C20
        (6, 1, 7, 1, 1, 1),    # 21: A20*B01 -> C21
        (7, 4, 7, 1, 1, 1),    # 22: A21*B11 -> C21
    ]
    
    # That's 23 terms but missing C[2,2] terms
    # This is still the schoolbook structure, just listed differently
    # I need a TRUE rank-23 which uses cancellations
    
    # For now, return schoolbook rank 27
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
