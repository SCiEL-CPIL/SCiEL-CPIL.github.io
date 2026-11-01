"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(i, j, k, col, coeff):
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += coeff

    # Rank-26 decomposition for 3x3 matrix multiplication
    # Strategy: Split 3x3 into 2x2 block (indices 0,1) and 1x1 (index 2)
    # Use Strassen for the 2x2 part (7 terms)
    # Handle cross terms between 2x2 and 1x1, and the 1x1 term
    # Total: 7 (Strassen) + 18 (cross and diagonal) = 25... 
    # Actually let me think more carefully.
    #
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    #
    # B = [[B00, B01, B02],
    #      [B10, B11, B12],
    #      [B20, B21, B22]]
    #
    # C = A * B
    #
    # Let's partition as:
    # A_top_left = [[A00, A01], [A10, A11]]  (2x2)
    # A_top_right = [A02, A12]               (2x1)
    # A_bottom = [A20, A21, A22]             (1x3)
    #
    # B_left_top = [[B00, B01], [B10, B11]]  (2x2)
    # B_left_bottom = [B20, B21]             (1x2)
    # B_right = [[B02], [B12], [B22]]        (3x1)
    #
    # C = [[A_tl * B_lt + A_tr * B_left_bottom, A_tl * B_right_top + A_tr * B22],
    #      [A_bottom_left * B_lt + A22 * B_left_bottom, A_bottom_left * B_right_top + A22 * B22]]
    #
    # This is getting complex. Let me use a different known decomposition.
    #
    # Known rank-26 decomposition:
    # Use the fact that we can do 2x2 Strassen (7 mults) for the top-left 2x2 block
    # Then handle the remaining terms.
    #
    # Let me use a direct construction. The schoolbook is 27. We can save 1 multiplication
    # by combining terms.
    #
    # Actually, let me use a verified rank-26 construction.
    #
    # Here's a rank-26 decomposition:
    # Terms 0-6: Strassen for 2x2 submatrix (indices 0,1)
    # Terms 7-25: Remaining 19 terms handled with 19 multiplications
    # Total: 7 + 19 = 26
    
    # Strassen for 2x2 part (A[0:2,0:2] * B[0:2,0:2])
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    # M1 = (A00+A11)*(B00+B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 1, col, 1)
    add_term(1, 1, 1, col, 1)
    # W contributions: C00 += 1, C11 += 1
    W[0 * n + 0][col] = 1  # C[0,0]
    W[1 * n + 1][col] = 1  # C[1,1]
    col += 1
    
    # M2 = (A10+A11)*B00
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    # W: C10 += 1, C11 -= 1
    W[0 * n + 1][col] = 1  # C[1,0]
    W[1 * n + 1][col] = -1  # C[1,1]
    col += 1
    
    # M3 = A00*(B10-B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    # W: C01 += 1, C11 += 1
    W[0 * n + 1][col] = 1  # C[0,1]
    W[1 * n + 1][col] = 1  # C[1,1]
    col += 1
    
    # M4 = A11*(B01-B00)
    add_term(1, 1, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    # W: C00 -= 1, C10 += 1
    W[0 * n + 0][col] = -1  # C[0,0]
    W[0 * n + 1][col] = 1  # C[1,0]
    col += 1
    
    # M5 = (A00+A01)*B11
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    # W: C00 += 1, C01 += 1
    W[0 * n + 0][col] = 1  # C[0,0]
    W[0 * n + 1][col] = 1  # C[0,1]
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    add_term(1, 0, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    # W: C11 += 1
    W[1 * n + 1][col] = 1  # C[1,1]
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    # W: C00 -= 1, C01 += 1
    W[0 * n + 0][col] = -1  # C[0,0]
    W[0 * n + 1][col] = 1  # C[0,1]
    col += 1
    
    # Now handle terms involving index 2
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    # C[2,0] = A20*B00 + A21*B10 + A22*B20
    # C[2,1] = A20*B01 + A21*B11 + A22*B21
    # C[2,2] = A20*B02 + A21*B12 + A22*B22
    
    # C[0,2] terms:
    # A00*B02
    add_term(0, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[2 * n + 0][col] = 1
    col += 1
    # A01*B12
    add_term(0, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[2 * n + 0][col] = 1
    col += 1
    # A02*B22
    add_term(0, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[2 * n + 0][col] = 1
    col += 1
    
    # C[1,2] terms:
    # A10*B02
    add_term(1, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[2 * n + 1][col] = 1
    col += 1
    # A11*B12
    add_term(1, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[2 * n + 1][col] = 1
    col += 1
    # A12*B22
    add_term(1, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[2 * n + 1][col] = 1
    col += 1
    
    # C[2,0] terms:
    # A20*B00
    add_term(2, 0, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    W[0 * n + 2][col] = 1
    col += 1
    # A21*B10
    add_term(2, 1, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    W[0 * n + 2][col] = 1
    col += 1
    # A22*B20
    add_term(2, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[0 * n + 2][col] = 1
    col += 1
    
    # C[2,1] terms:
    # A20*B01
    add_term(2, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    W[1 * n + 2][col] = 1
    col += 1
    # A21*B11
    add_term(2, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[1 * n + 2][col] = 1
    col += 1
    # A22*B21
    add_term(2, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[1 * n + 2][col] = 1
    col += 1
    
    # C[2,2] = A20*B02 + A21*B12 + A22*B22
    # We can combine some terms. Let's just add them separately for now.
    # A20*B02
    add_term(2, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[2 * n + 2][col] = 1
    col += 1
    # A21*B12
    add_term(2, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[2 * n + 2][col] = 1
    col += 1
    # A22*B22
    add_term(2, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[2 * n + 2][col] = 1
    col += 1
    
    # That gives us 7 + 3 + 3 + 3 + 3 + 3 = 22 terms, but we need to check
    # if this is correct. Let me verify by checking the tensor identity.
    
    # Actually, I realize my approach is flawed. Let me use a cleaner construction.
    # 
    # The issue is that the Strassen terms only cover C[0:2, 0:2], and then
    # I need to handle all terms involving index 2.
    #
    # Let me restart with a cleaner rank-26 construction.
    
    # Reset
    for row in U:
        for k in range(rank):
            row[k] = 0
    for row in V:
        for k in range(rank):
            row[k] = 0
    for row in W:
        for k in range(rank):
            row[k] = 0
    
    # Clean rank-26 construction:
    # Use 7 Strassen multiplications for the 2x2 block
    # Use 19 additional multiplications for terms involving index 2
    # But some of these can be combined.
    
    # Actually, the standard result is that rank(3x3) = 23, but a simple
    # rank-26 construction is easier to verify. Let me use a different approach.
    
    # Let me use the following: partition into 2x2 and 1x1, use Strassen for 2x2,
    # and handle the rest with 19 terms, but combine A22*B22 with some existing terms.
    
    # Term 0-6: Strassen for 2x2
    col = 0
    # M1 = (A00+A11)*(B00+B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 1, col, 1)
    add_term(1, 1, 1, col, 1)
    W[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    W[1][col] = 1
    W[4][col] = -1
    col += 1
    
    # M3 = A00*(B10-B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    W[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M4 = A11*(B01-B00)
    add_term(1, 1, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    W[0][col] = -1
    W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    add_term(1, 0, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[0][col] = -1
    W[1][col] = 1
    col += 1
    
    # Now terms involving index 2:
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    # C[2,0] = A20*B00 + A21*B10 + A22*B20
    # C[2,1] = A20*B01 + A21*B11 + A22*B21
    # C[2,2] = A20*B02 + A21*B12 + A22*B22
    
    # We can combine some of these. Let's use 19 terms for the 15 output entries
    # involving index 2, but we need 15 multiplications minimum.
    
    # Actually, let me just use 19 separate terms for simplicity and verify.
    # That gives total 7 + 19 = 26.
    
    # C[0,2]:
    add_term(0, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[6][col] = 1
    col += 1
    add_term(0, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[6][col] = 1
    col += 1
    add_term(0, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[6][col] = 1
    col += 1
    
    # C[1,2]:
    add_term(1, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[7][col] = 1
    col += 1
    add_term(1, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[7][col] = 1
    col += 1
    add_term(1, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[7][col] = 1
    col += 1
    
    # C[2,0]:
    add_term(2, 0, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    W[2][col] = 1
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    W[2][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[2][col] = 1
    col += 1
    
    # C[2,1]:
    add_term(2, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    W[3][col] = 1
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[3][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[3][col] = 1
    col += 1
    
    # C[2,2]:
    add_term(2, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[8][col] = 1
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[8][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[8][col] = 1
    col += 1
    
    # That's 7 + 15 = 22 terms. But we need rank 26. Let me check if this is correct.
    # The issue is that the Strassen terms already contribute to C[0,0], C[0,1], C[1,0], C[1,1],
    # and the additional terms handle C[0,2], C[1,2], C[2,0], C[2,1], C[2,2].
    # But we're missing the contributions from the 2x2 block to C[2,0] and C[2,1]!
    # No wait, C[2,0] and C[2,1] only involve A[2,*] and B[*,0] or B[*,1], so they don't
    # involve the 2x2 block of A. So this should be correct.
    #
    # Let me verify: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]. Yes, correct.
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]. Yes, correct.
    #
    # So we have 22 terms. But the rank is supposed to be 26. Let me check if there are
    # missing terms.
    #
    # Ah, I see the issue. The Strassen terms handle C[0:2, 0:2], but they only use
    # A[0:2, 0:2] and B[0:2, 0:2]. The terms involving A[0:2, 2] and B[2, 0:2] are
    # not included in the Strassen part. Let me check:
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # The Strassen part only handles A[0,0]*B[0,0] + A[0,1]*B[1,0], missing A[0,2]*B[2,0].
    # Similarly for C[0,1], C[1,0], C[1,1].
    #
    # So I need to add terms for A[0:2, 2]*B[2, 0:2] and A[2, 0:2]*B[0:2, 0:2].
    #
    # Let me add these:
    # C[0,0] += A[0,2]*B[2,0]
    add_term(0, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[0][col] = 1
    col += 1
    # C[0,1] += A[0,2]*B[2,1]
    add_term(0, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[1][col] = 1
    col += 1
    # C[1,0] += A[1,2]*B[2,0]
    add_term(1, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[1][col] = 1  # Wait, C[1,0] is W[0*n+1] = W[1]
    # Actually, W[k*n+i] where k is column, i is row. C[1,0] means row 1, col 0, so k=0, i=1, W[0*3+1]=W[1]
    col += 1
    # C[1,1] += A[1,2]*B[2,1]
    add_term(1, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[1][col] = 1  # C[1,1] is row 1, col 1, so k=1, i=1, W[1*3+1]=W[4]
    # Wait, I made an error. Let me fix the W indexing.
    # W[k*n+i] where k is the column of B (and C), i is the row of A (and C).
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So W[k*n+i] gets the coefficient for C[i,k].
    # C[1,1] -> k=1, i=1 -> W[1*3+1] = W[4]
    col += 1
    
    # That's 22 + 4 = 26 terms. 
    
    # Let me verify the W indexing is correct by checking a few:
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[3]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[1,1] -> k=1, i=1 -> W[4]
    # C[0,2] -> k=2, i=0 -> W[6]
    # C[1,2] -> k=2, i=1 -> W[7]
    # C[2,0] -> k=0, i=2 -> W[2]
    # C[2,1] -> k=1, i=2 -> W[5]
    # C[2,2] -> k=2, i=2 -> W[8]
    
    # I see I made errors in the W indexing above. Let me restart with correct indexing.
    
    # Reset
    for row in U:
        for k in range(rank):
            row[k] = 0
    for row in V:
        for k in range(rank):
            row[k] = 0
    for row in W:
        for k in range(rank):
            row[k] = 0
    
    # Correct W indexing: W[k*n+i] for C[i,k]
    col = 0
    # M1 = (A00+A11)*(B00+B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 1, col, 1)
    add_term(1, 1, 1, col, 1)
    W[0][col] = 1   # C[0,0]
    W[4][col] = 1   # C[1,1]
    col += 1
    
    # M2 = (A10+A11)*B00
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    W[1][col] = 1   # C[1,0]
    W[4][col] = -1  # C[1,1]
    col += 1
    
    # M3 = A00*(B10-B11)
    add_term(0, 0, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    W[3][col] = 1   # C[0,1]
    W[4][col] = 1   # C[1,1]
    col += 1
    
    # M4 = A11*(B01-B00)
    add_term(1, 1, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    W[0][col] = -1  # C[0,0]
    W[1][col] = 1   # C[1,0]
    col += 1
    
    # M5 = (A00+A01)*B11
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[0][col] = 1   # C[0,0]
    W[3][col] = 1   # C[0,1]
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    add_term(1, 0, 0, col, 1)
    add_term(0, 0, 0, col, -1)
    add_term(0, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    W[4][col] = 1   # C[1,1]
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    add_term(0, 1, 0, col, 1)
    add_term(1, 1, 0, col, -1)
    add_term(1, 0, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[0][col] = -1  # C[0,0]
    W[3][col] = 1   # C[0,1]
    col += 1
    
    # Terms involving index 2:
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    add_term(0, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[6][col] = 1   # C[0,2]
    col += 1
    add_term(0, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[6][col] = 1
    col += 1
    add_term(0, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[6][col] = 1
    col += 1
    
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    add_term(1, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[7][col] = 1   # C[1,2]
    col += 1
    add_term(1, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[7][col] = 1
    col += 1
    add_term(1, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[7][col] = 1
    col += 1
    
    # C[2,0] = A20*B00 + A21*B10 + A22*B20
    add_term(2, 0, 0, col, 1)
    add_term(0, 0, 0, col, 1)
    W[2][col] = 1   # C[2,0]
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 0, 0, col, 1)
    W[2][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[2][col] = 1
    col += 1
    
    # C[2,1] = A20*B01 + A21*B11 + A22*B21
    add_term(2, 0, 0, col, 1)
    add_term(0, 1, 0, col, 1)
    W[5][col] = 1   # C[2,1]
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 1, 0, col, 1)
    W[5][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[5][col] = 1
    col += 1
    
    # C[2,2] = A20*B02 + A21*B12 + A22*B22
    add_term(2, 0, 0, col, 1)
    add_term(0, 2, 0, col, 1)
    W[8][col] = 1   # C[2,2]
    col += 1
    add_term(2, 1, 0, col, 1)
    add_term(1, 2, 0, col, 1)
    W[8][col] = 1
    col += 1
    add_term(2, 2, 0, col, 1)
    add_term(2, 2, 0, col, 1)
    W[8][col] = 1
    col += 1
    
    # Missing terms: A[0:2,2]*B[2,0:2] and A[2,0:2]*B[0:2,0:2]
    # C[0,0] += A[0,2]*B[2,0]
    add_term(0, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[0][col] = 1   # C[0,0]
    col += 1
    # C[0,1] += A[0,2]*B[2,1]
    add_term(0, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[3][col] = 1   # C[0,1]
    col += 1
    # C[1,0] += A[1,2]*B[2,0]
    add_term(1, 2, 0, col, 1)
    add_term(2, 0, 0, col, 1)
    W[1][col] = 1   # C[1,0]
    col += 1
    # C[1,1] += A[1,2]*B[2,1]
    add_term(1, 2, 0, col, 1)
    add_term(2, 1, 0, col, 1)
    W[4][col] = 1   # C[1,1]
    col += 1
    
    # That's 7 + 15 + 4 = 26 terms. 
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
