"""Strassen-based tensor decomposition for 2x3 times 3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen on the 2x2 part (n=2, and 2 of the 3 in m)
    # The remaining dimension is 1 (in m) and 2 (in p) for the tail
    # Total rank: 3*7 + 1*1*2 = 21 + 2 = 23
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]  # 6 x 23
    V = [[0] * rank for _ in range(m * p)]  # 18 x 23
    W = [[0] * rank for _ in range(p * n)]  # 12 x 23
    
    # U row index: i*m + j, where i in 0..1, j in 0..2
    # V row index: j*p + k, where j in 0..2, k in 0..5
    # W row index: k*n + i, where k in 0..5, i in 0..1
    
    # Strassen 7-term decomposition for 2x2 blocks
    # For each block b in 0..2 (corresponding to 2 of the 3 columns in m, 2 of the 6 in p)
    # Let's say we group m as: first 2 dimensions use Strassen, last 1 is tail
    # And p as: for each pair of m dimensions, we use 2 p dimensions
    # Actually, let me think more carefully.
    
    # n=2, m=3, p=6
    # We can view this as: for each of the 3 "columns" of A (each of size 2),
    # we multiply with the corresponding 2x6 block of B.
    # But Strassen works on 2x2 matrices.
    
    # Let me reconsider. We have A (2x3) and B (3x6).
    # We can split m=3 into 2+1, and p=6 into 2*3 (one 2-wide strip per m-dimension).
    # For the first 2 m-dimensions, we can use Strassen on the 2x2 x 2x2 blocks.
    # For the last m-dimension, we use schoolbook on the 2x1 x 1x2 block.
    
    # Actually, a cleaner approach:
    # Split m into two groups: {0,1} and {2}
    # For group {0,1}: use Strassen on A[:, :2] (2x2) and B[:2, :] (2x6)
    #   But Strassen needs square matrices. We can split B[:2, :] into three 2x2 blocks.
    #   For each 2x2 block, use 7 Strassen terms.
    # For group {2}: use schoolbook on A[:, 2] (2x1) and B[2, :] (1x6)
    #   That's 2*1*6 = 12 schoolbook terms.
    
    # Total: 3 blocks * 7 Strassen + 12 schoolbook = 21 + 12 = 33.
    # That's what the measured improvement got.
    
    # Can we do better? Let's try to also use Strassen on the tail.
    # The tail is 2x1 times 1x6. We can split p=6 into three 2-wide strips.
    # For each strip, it's 2x1 times 1x2. That's rank 4 schoolbook, or can we do better?
    # 2x1 x 1x2 tensor has rank 4 (it's just outer product structure, can't be compressed).
    # So 3 strips * 4 = 12. Same as before.
    
    # What if we group differently? Split m into {0,2} and {1}?
    # Or use a different factorization?
    
    # Let me just implement the 33-rank Strassen decomposition and see if it's valid.
    
    # Strassen terms for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - B00)
    # M5 = (A00 + A10)(B11)
    # M6 = (A01 - A11)(B00 + B10)
    # M7 = (A01 - A10)(B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem:
    # A is 2x3, B is 3x6.
    # Let's split A into A_left (2x2, columns 0,1) and A_right (2x1, column 2).
    # Let's split B into B_top (2x6, rows 0,1) and B_bottom (1x6, row 2).
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top:
    #   B_top is 2x6, split into three 2x2 blocks: B_top[:,:2], B_top[:,2:4], B_top[:,4:6]
    #   For each block, use Strassen (7 terms each)
    
    # For A_right @ B_bottom:
    #   A_right is 2x1, B_bottom is 1x6
    #   Split B_bottom into three 1x2 blocks
    #   For each block, it's a 2x1 x 1x2 tensor, rank 4
    #   Total: 3 * 4 = 12 terms
    
    # Grand total: 3 * 7 + 12 = 33
    
    # Let me verify the indices:
    # U[i*m + j, c] where i=0,1 and j=0,1,2
    # V[j*p + k, c] where j=0,1,2 and k=0,1,2,3,4,5
    # W[k*n + i, c] where k=0,1,2,3,4,5 and i=0,1
    
    # Strassen block b (b=0,1,2), using columns 2b, 2b+1 of B_top
    # A_left rows: i=0,1; columns: j=0,1
    # B_top rows: j=0,1; columns: k=2b, 2b+1
    
    # For Strassen term t in 0..6:
    #   A coefficients:
    #     M1: A00 + A11 -> U[0*3+0]=U[0] +=1, U[1*3+1]=U[4] +=1
    #     M2: A10 + A11 -> U[1*3+0]=U[3] +=1, U[1*3+1]=U[4] +=1
    #     M3: A00 -> U[0] +=1
    #     M4: A11 -> U[4] +=1
    #     M5: A00 + A10 -> U[0] +=1, U[3] +=1
    #     M6: A01 - A11 -> U[1] +=1, U[4] -=1
    #     M7: A01 - A10 -> U[1] +=1, U[3] -=1
    
    #   B coefficients (for block b, rows j=0,1, cols k=2b, 2b+1):
    #     B00 = B[0, 2b] -> V[0*6+2b] = V[2b]
    #     B01 = B[0, 2b+1] -> V[2b+1]
    #     B10 = B[1, 2b] -> V[6+2b] = V[6+2b]
    #     B11 = B[1, 2b+1] -> V[6+2b+1] = V[7+2b]
    
    #     M1: B00 + B11 -> V[2b] +=1, V[7+2b] +=1
    #     M2: B00 -> V[2b] +=1
    #     M3: B01 - B11 -> V[2b+1] +=1, V[7+2b] -=1
    #     M4: B10 - B00 -> V[6+2b] +=1, V[2b] -=1
    #     M5: B11 -> V[7+2b] +=1
    #     M6: B00 + B10 -> V[2b] +=1, V[6+2b] +=1
    #     M7: B01 + B11 -> V[2b+1] +=1, V[7+2b] +=1
    
    #   C coefficients (output indices k=2b,2b+1; i=0,1):
    #     W[k*n + i] = W[2b*2 + i] = W[4b+i] for k=2b, i=0,1
    #     W[(2b+1)*2 + i] = W[4b+2+i] for k=2b+1, i=0,1
    #     So: C00 (i=0,k=2b) -> W[4b], C01 (i=0,k=2b+1) -> W[4b+1]
    #         C10 (i=1,k=2b) -> W[4b+2], C11 (i=1,k=2b+1) -> W[4b+3]
    
    #     M1: C00 +1, C01 +1, C10 +1, C11 -1? No:
    #     C00 = M1 + M4 - M5 + M7
    #     C01 = M3 + M5
    #     C10 = M2 + M4
    #     C11 = M1 - M2 + M3 + M6
    
    #     So:
    #     M1: W[4b] +=1, W[4b+3] +=1
    #     M2: W[4b+2] +=1, W[4b+3] -=1
    #     M3: W[4b+1] +=1, W[4b+3] +=1
    #     M4: W[4b] +=1, W[4b+2] +=1
    #     M5: W[4b] -=1, W[4b+1] +=1
    #     M6: W[4b+3] +=1
    #     M7: W[4b] +=1
    
    # Let me code this up.
    
    col = 0
    for b in range(3):
        for t in range(7):
            # A coefficients
            if t == 0:  # M1
                U[0][col] += 1
                U[4][col] += 1
            elif t == 1:  # M2
                U[3][col] += 1
                U[4][col] += 1
            elif t == 2:  # M3
                U[0][col] += 1
            elif t == 3:  # M4
                U[4][col] += 1
            elif t == 4:  # M5
                U[0][col] += 1
                U[3][col] += 1
            elif t == 5:  # M6
                U[1][col] += 1
                U[4][col] -= 1
            elif t == 6:  # M7
                U[1][col] += 1
                U[3][col] -= 1
            
            # B coefficients
            if t == 0:  # M1
                V[2*b][col] += 1
                V[7+2*b][col] += 1
            elif t == 1:  # M2
                V[2*b][col] += 1
            elif t == 2:  # M3
                V[2*b+1][col] += 1
                V[7+2*b][col] -= 1
            elif t == 3:  # M4
                V[6+2*b][col] += 1
                V[2*b][col] -= 1
            elif t == 4:  # M5
                V[7+2*b][col] += 1
            elif t == 5:  # M6
                V[2*b][col] += 1
                V[6+2*b][col] += 1
            elif t == 6:  # M7
                V[2*b+1][col] += 1
                V[7+2*b][col] += 1
            
            # C coefficients
            if t == 0:  # M1
                W[4*b][col] += 1
                W[4*b+3][col] += 1
            elif t == 1:  # M2
                W[4*b+2][col] += 1
                W[4*b+3][col] -= 1
            elif t == 2:  # M3
                W[4*b+1][col] += 1
                W[4*b+3][col] += 1
            elif t == 3:  # M4
                W[4*b][col] += 1
                W[4*b+2][col] += 1
            elif t == 4:  # M5
                W[4*b][col] -= 1
                W[4*b+1][col] += 1
            elif t == 5:  # M6
                W[4*b+3][col] += 1
            elif t == 6:  # M7
                W[4*b][col] += 1
            
            col += 1
    
    # Now the tail: A_right (2x1) @ B_bottom (1x6)
    # A_right: column 2 of A, so U rows: i*3+2 for i=0,1 -> U[2], U[5]
    # B_bottom: row 2 of B, so V rows: 2*6+k = 12+k for k=0..5
    # Output: W[k*2+i] for k=0..5, i=0,1
    
    # For each k in 0..5:
    #   U[2][col] += 1  (A[0,2])
    #   V[12+k][col] += 1  (B[2,k])
    #   W[2*k+0][col] += 1  (C[0,k])
    #   U[5][col] += 1  (A[1,2])
    #   V[12+k][col] += 1  (B[2,k])
    #   W[2*k+1][col] += 1  (C[1,k])
    
    # But we can combine: for each k, use 2 terms (one for each i)
    # Or use 4 terms per 2x1 x 1x2 block.
    
    # Actually, the 2x1 x 1x2 tensor T[i,j,k] = A[i] * B[j] where j is fixed.
    # It decomposes as: for each i, term: U[i*3+2]=1, V[12+k]=1, W[2k+i]=1
    # That's 2 terms per k, total 12 terms.
    
    for k in range(6):
        for i in range(2):
            U[i*3+2][col] = 1
            V[12+k][col] = 1
            W[2*k+i][col] = 1
            col += 1
    
    # col should be 21 + 12 = 33
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
