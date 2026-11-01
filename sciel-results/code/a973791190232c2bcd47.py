"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition: Strassen on 2x2 block + schoolbook on remaining 1x3 / 3x1 / 1x1
    # Matrices are partitioned as:
    #   A = [[A11, A12],   B = [[B11, B12],
    #        [A21, A22]]         [B21, B22]]
    # where A11, B11 are 2x2, A12 and B21 are 2x1, A21 and B12 are 1x2, A22 and B22 are 1x1.
    #
    # C11 = A11*B11 (Strassen: 7 terms)
    # C12 = A11*B12 + A12*B22 (2x1 * 1x2 outer + 2x1 * 1x1 = 4 terms? No, A11*B12 is 2x2*1x2 = 2x2, 4 terms; A12*B22 is 2x1*1x1 = 2 terms. Total 6)
    # Wait, let's be more careful with dimensions.
    # A is 3x3, B is 3x3.
    # A11: 2x2, A12: 2x1, A21: 1x2, A22: 1x1
    # B11: 2x2, B12: 2x1, B21: 1x2, B22: 1x1
    #
    # C11 (2x2) = A11 * B11  -> Strassen 7 terms
    # C12 (2x1) = A11 * B12 + A12 * B22
    #   A11 (2x2) * B12 (2x1) = 2x1, 4 terms (schoolbook)
    #   A12 (2x1) * B22 (1x1) = 2x1, 2 terms
    #   Total: 6 terms
    #
    # C21 (1x2) = A21 * B11 + A22 * B21
    #   A21 (1x2) * B11 (2x2) = 1x2, 4 terms
    #   A22 (1x1) * B21 (1x2) = 1x2, 2 terms
    #   Total: 6 terms
    #
    # C22 (1x1) = A21 * B12 + A22 * B22
    #   A21 (1x2) * B12 (2x1) = 1x1, 2 terms
    #   A22 (1x1) * B22 (1x1) = 1x1, 1 term
    #   Total: 3 terms
    #
    # Grand total: 7 + 6 + 6 + 3 = 22 terms.
    #
    # Wait, the knowledge said rank-26. Let me re-read.
    # "Replaces the rank-27 schoolbook decomposition with a rank-26 decomposition by partitioning the 3x3 matrices into 2x2 and 1x1 blocks and applying Strassen's algorithm to the 2x2 block, reducing the rank by 1."
    #
    # My calculation gives 22. Why the discrepancy?
    # Ah, the "1x1 blocks" phrasing might mean partitioning into 2x2, 2x1, 1x2, 1x1.
    # My calculation is 22. Is 22 valid?
    # Let's verify the Strassen part.
    # C11 = A11 * B11.
    # M1 = (A11[0,0] + A11[1,1]) * (B11[0,0] + B11[1,1])
    # M2 = (A11[1,0] + A11[1,1]) * B11[0,0]
    # M3 = A11[0,0] * (B11[0,1] - B11[1,1])
    # M4 = A11[1,1] * (B11[1,0] - B11[0,0])
    # M5 = (A11[0,0] + A11[0,1]) * B11[1,1]
    # M6 = (A11[1,0] - A11[0,0]) * (B11[0,0] + B11[0,1])
    # M7 = (A11[0,1] - A11[1,1]) * (B11[1,0] + B11[1,1])
    #
    # C11[0,0] = M1 + M4 - M5 + M7
    # C11[0,1] = M3 + M5
    # C11[1,0] = M2 + M4
    # C11[1,1] = M1 - M2 + M3 + M6
    #
    # These are 7 multiplications. The additions/subtractions are free in this context (they are just linear combinations of the inputs in U, V, W).
    # So yes, 7 terms for C11.
    #
    # C12 (2x1):
    # C12[0,0] = A11[0,0]*B12[0,0] + A11[0,1]*B12[1,0] + A12[0,0]*B22[0,0]
    # C12[1,0] = A11[1,0]*B12[0,0] + A11[1,1]*B12[1,0] + A12[1,0]*B22[0,0]
    # 4 terms from A11*B12, 2 terms from A12*B22. Total 6.
    #
    # C21 (1x2):
    # C21[0,0] = A21[0,0]*B11[0,0] + A21[0,1]*B11[1,0] + A22[0,0]*B21[0,0]
    # C21[0,1] = A21[0,0]*B11[0,1] + A21[0,1]*B11[1,1] + A22[0,0]*B21[0,1]
    # 4 terms from A21*B11, 2 terms from A22*B21. Total 6.
    #
    # C22 (1x1):
    # C22[0,0] = A21[0,0]*B12[0,0] + A21[0,1]*B12[1,0] + A22[0,0]*B22[0,0]
    # 2 terms from A21*B12, 1 term from A22*B22. Total 3.
    #
    # Total: 7 + 6 + 6 + 3 = 22.
    #
    # So I can achieve rank 22. This is better than 26.
    #
    # Let's construct the U, V, W matrices.
    # U shape: (n*m, rank) = (9, 22)
    # V shape: (m*p, rank) = (9, 22)
    # W shape: (p*n, rank) = (9, 22)
    #
    # Indexing convention:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    #
    # n=3, m=3, p=3.
    #
    # Let's list the 22 terms.
    # Term 0-6: Strassen for C11
    # Term 7-10: A11*B12
    # Term 11-12: A12*B22
    # Term 13-16: A21*B11
    # Term 17-18: A22*B21
    # Term 19-20: A21*B12
    # Term 21: A22*B22
    #
    # Let's define the indices.
    # A11 = A[0:2, 0:2]
    # A12 = A[0:2, 2]
    # A21 = A[2, 0:2]
    # A22 = A[2, 2]
    #
    # B11 = B[0:2, 0:2]
    # B12 = B[0:2, 2]
    # B21 = B[2, 0:2]
    # B22 = B[2, 2]
    #
    # C11 = C[0:2, 0:2]
    # C12 = C[0:2, 2]
    # C21 = C[2, 0:2]
    # C22 = C[2, 2]
    #
    # Strassen Terms for C11 (indices 0-6):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's map these to U, V, W.
    # U[i*3 + j, c], V[j*3 + k, c], W[k*3 + i, c]
    #
    # Term 0 (M1):
    # U: A00=1, A11=1 -> U[0*3+0]=1, U[1*3+1]=1
    # V: B00=1, B11=1 -> V[0*3+0]=1, V[1*3+1]=1
    # W: C00=1, C11=1 -> W[0*3+0]=1, W[1*3+1]=1
    #
    # Term 1 (M2):
    # U: A10=1, A11=1 -> U[1*3+0]=1, U[1*3+1]=1
    # V: B00=1 -> V[0*3+0]=1
    # W: C10=1 -> W[0*3+1]=1
    #
    # Term 2 (M3):
    # U: A00=1 -> U[0*3+0]=1
    # V: B01=1, B11=-1 -> V[0*3+1]=1, V[1*3+1]=-1
    # W: C01=1 -> W[1*3+0]=1
    #
    # Term 3 (M4):
    # U: A11=1 -> U[1*3+1]=1
    # V: B10=1, B00=-1 -> V[1*3+0]=1, V[0*3+0]=-1
    # W: C00=1, C10=1 -> W[0*3+0]=1, W[0*3+1]=1
    #
    # Term 4 (M5):
    # U: A00=1, A01=1 -> U[0*3+0]=1, U[0*3+1]=1
    # V: B11=1 -> V[1*3+1]=1
    # W: C00=-1, C01=1 -> W[0*3+0]=-1, W[1*3+0]=1
    #
    # Term 5 (M6):
    # U: A10=1, A00=-1 -> U[1*3+0]=1, U[0*3+0]=-1
    # V: B00=1, B01=1 -> V[0*3+0]=1, V[0*3+1]=1
    # W: C11=1 -> W[1*3+1]=1
    #
    # Term 6 (M7):
    # U: A01=1, A11=-1 -> U[0*3+1]=1, U[1*3+1]=-1
    # V: B10=1, B11=1 -> V[1*3+0]=1, V[1*3+1]=1
    # W: C00=1, C11=1 -> W[0*3+0]=1, W[1*3+1]=1
    #
    # Let's double check Term 4 (M5).
    # C00 = ... - M5. So W[0,0] for Term 4 should be -1.
    # C01 = M3 + M5. So W[1,0] for Term 4 should be 1.
    # Correct.
    #
    # Term 5 (M6).
    # C11 = ... + M6. So W[1,1] for Term 5 should be 1.
    # Correct.
    #
    # Term 6 (M7).
    # C00 = ... + M7. So W[0,0] for Term 6 should be 1.
    # C11 = ... + M7. So W[1,1] for Term 6 should be 1.
    # Correct.
    #
    # Now the schoolbook parts.
    # Term 7-10: A11 * B12 -> C12
    # C12 is 2x1. Indices in C: (0,2), (1,2).
    # A11 is 2x2. B12 is 2x1.
    # C12[0] = A11[0,0]*B12[0] + A11[0,1]*B12[1]
    # C12[1] = A11[1,0]*B12[0] + A11[1,1]*B12[1]
    #
    # Term 7: A11[0,0] * B12[0] -> C12[0]
    # U: U[0*3+0]=1
    # V: V[0*3+2]=1  (B12[0] is B[0,2])
    # W: W[2*3+0]=1  (C12[0] is C[0,2], k=2, i=0 -> W[2*3+0])
    #
    # Term 8: A11[0,1] * B12[1] -> C12[0]
    # U: U[0*3+1]=1
    # V: V[1*3+2]=1  (B12[1] is B[1,2])
    # W: W[2*3+0]=1
    #
    # Term 9: A11[1,0] * B12[0] -> C12[1]
    # U: U[1*3+0]=1
    # V: V[0*3+2]=1
    # W: W[2*3+1]=1  (C12[1] is C[1,2], k=2, i=1 -> W[2*3+1])
    #
    # Term 10: A11[1,1] * B12[1] -> C12[1]
    # U: U[1*3+1]=1
    # V: V[1*3+2]=1
    # W: W[2*3+1]=1
    #
    # Term 11-12: A12 * B22 -> C12
    # A12 is 2x1. B22 is 1x1.
    # C12[0] += A12[0] * B22
    # C12[1] += A12[1] * B22
    #
    # Term 11: A12[0] * B22 -> C12[0]
    # U: U[0*3+2]=1  (A12[0] is A[0,2])
    # V: V[2*3+2]=1  (B22 is B[2,2])
    # W: W[2*3+0]=1
    #
    # Term 12: A12[1] * B22 -> C12[1]
    # U: U[1*3+2]=1  (A12[1] is A[1,2])
    # V: V[2*3+2]=1
    # W: W[2*3+1]=1
    #
    # Term 13-16: A21 * B11 -> C21
    # A21 is 1x2. B11 is 2x2.
    # C21 is 1x2. Indices in C: (2,0), (2,1).
    # C21[0] = A21[0]*B11[0,0] + A21[1]*B11[1,0]
    # C21[1] = A21[0]*B11[0,1] + A21[1]*B11[1,1]
    #
    # Term 13: A21[0] * B11[0,0] -> C21[0]
    # U: U[2*3+0]=1  (A21[0] is A[2,0])
    # V: V[0*3+0]=1
    # W: W[0*3+2]=1  (C21[0] is C[2,0], k=0, i=2 -> W[0*3+2])
    #
    # Term 14: A21[1] * B11[1,0] -> C21[0]
    # U: U[2*3+1]=1  (A21[1] is A[2,1])
    # V: V[1*3+0]=1
    # W: W[0*3+2]=1
    #
    # Term 15: A21[0] * B11[0,1] -> C21[1]
    # U: U[2*3+0]=1
    # V: V[0*3+1]=1
    # W: W[1*3+2]=1  (C21[1] is C[2,1], k=1, i=2 -> W[1*3+2])
    #
    # Term 16: A21[1] * B11[1,1] -> C21[1]
    # U: U[2*3+1]=1
    # V: V[1*3+1]=1
    # W: W[1*3+2]=1
    #
    # Term 17-18: A22 * B21 -> C21
    # A22 is 1x1. B21 is 1x2.
    # C21[0] += A22 * B21[0]
    # C21[1] += A22 * B21[1]
    #
    # Term 17: A22 * B21[0] -> C21[0]
    # U: U[2*3+2]=1  (A22 is A[2,2])
    # V: V[2*3+0]=1  (B21[0] is B[2,0])
    # W: W[0*3+2]=1
    #
    # Term 18: A22 * B21[1] -> C21[1]
    # U: U[2*3+2]=1
    # V: V[2*3+1]=1
    # W: W[1*3+2]=1
    #
    # Term 19-20: A21 * B12 -> C22
    # A21 is 1x2. B12 is 2x1.
    # C22 is 1x1. Index in C: (2,2).
    # C22 = A21[0]*B12[0] + A21[1]*B12[1]
    #
    # Term 19: A21[0] * B12[0] -> C22
    # U: U[2*3+0]=1
    # V: V[0*3+2]=1
    # W: W[2*3+2]=1  (C22 is C[2,2], k=2, i=2 -> W[2*3+2])
    #
    # Term 20: A21[1] * B12[1] -> C22
    # U: U[2*3+1]=1
    # V: V[1*3+2]=1
    # W: W[2*3+2]=1
    #
    # Term 21: A22 * B22 -> C22
    # U: U[2*3+2]=1
    # V: V[2*3+2]=1
    # W: W[2*3+2]=1
    #
    # Total 22 terms.
    #
    # Let's build the matrices.
    n, m, p = 3, 3, 3
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1):
        pass # We will do it manually for clarity and control
    
    # Term 0 (M1)
    U[0*3+0][0] += 1; U[1*3+1][0] += 1
    V[0*3+0][0] += 1; V[1*3+1][0] += 1
    W[0*3+0][0] += 1; W[1*3+1][0] += 1
    
    # Term 1 (M2)
    U[1*3+0][1] += 1; U[1*3+1][1] += 1
    V[0*3+0][1] += 1
    W[0*3+1][1] += 1
    
    # Term 2 (M3)
    U[0*3+0][2] += 1
    V[0*3+1][2] += 1; V[1*3+1][2] += -1
    W[1*3+0][2] += 1
    
    # Term 3 (M4)
    U[1*3+1][3] += 1
    V[1*3+0][3] += 1; V[0*3+0][3] += -1
    W[0*3+0][3] += 1; W[0*3+1][3] += 1
    
    # Term 4 (M5)
    U[0*3+0][4] += 1; U[0*3+1][4] += 1
    V[1*3+1][4] += 1
    W[0*3+0][4] += -1; W[1*3+0][4] += 1
    
    # Term 5 (M6)
    U[1*3+0][5] += 1; U[0*3+0][5] += -1
    V[0*3+0][5] += 1; V[0*3+1][5] += 1
    W[1*3+1][5] += 1
    
    # Term 6 (M7)
    U[0*3+1][6] += 1; U[1*3+1][6] += -1
    V[1*3+0][6] += 1; V[1*3+1][6] += 1
    W[0*3+0][6] += 1; W[1*3+1][6] += 1
    
    # Term 7
    U[0*3+0][7] += 1
    V[0*3+2][7] += 1
    W[2*3+0][7] += 1
    
    # Term 8
    U[0*3+1][8] += 1
    V[1*3+2][8] += 1
    W[2*3+0][8] += 1
    
    # Term 9
    U[1*3+0][9] += 1
    V[0*3+2][9] += 1
    W[2*3+1][9] += 1
    
    # Term 10
    U[1*3+1][10] += 1
    V[1*3+2][10] += 1
    W[2*3+1][10] += 1
    
    # Term 11
    U[0*3+2][11] += 1
    V[2*3+2][11] += 1
    W[2*3+0][11] += 1
    
    # Term 12
    U[1*3+2][12] += 1
    V[2*3+2][12] += 1
    W[2*3+1][12] += 1
    
    # Term 13
    U[2*3+0][13] += 1
    V[0*3+0][13] += 1
    W[0*3+2][13] += 1
    
    # Term 14
    U[2*3+1][14] += 1
    V[1*3+0][14] += 1
    W[0*3+2][14] += 1
    
    # Term 15
    U[2*3+0][15] += 1
    V[0*3+1][15] += 1
    W[1*3+2][15] += 1
    
    # Term 16
    U[2*3+1][16] += 1
    V[1*3+1][16] += 1
    W[1*3+2][16] += 1
    
    # Term 17
    U[2*3+2][17] += 1
    V[2*3+0][17] += 1
    W[0*3+2][17] += 1
    
    # Term 18
    U[2*3+2][18] += 1
    V[2*3+1][18] += 1
    W[1*3+2][18] += 1
    
    # Term 19
    U[2*3+0][19] += 1
    V[0*3+2][19] += 1
    W[2*3+2][19] += 1
    
    # Term 20
    U[2*3+1][20] += 1
    V[1*3+2][20] += 1
    W[2*3+2][20] += 1
    
    # Term 21
    U[2*3+2][21] += 1
    V[2*3+2][21] += 1
    W[2*3+2][21] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
