"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x3 * 3x8 using a 2+1 block decomposition
    with Strassen's algorithm for the 2x2 top-left block.
    
    Rank = 68 (vs schoolbook 72)
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    # U shape: (n*m, rank) = (9, 68)
    # V shape: (m*p, rank) = (24, 68)
    # W shape: (p*n, rank) = (24, 68)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Term 1: 4 Strassen 2x2 blocks (28 terms total)
    # For each 4-column block t in B (t = 0..3):
    #   A_block = top-left 2x2 of A: entries A[0][0], A[0][1], A[1][0], A[1][1]
    #   B_block = 2x4 slice of B: rows 0,1; cols 4t..4t+3
    #   C_block = 2x4 slice of C: rows 0,1; cols 4t..4t+3
    #
    # Strassen for 2x2 * 2x2 (7 multiplications):
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 + M7
    #
    # But we have 2x4, so we split B_block into two 2x2 halves.
    # Actually, Strassen works on 2x2 blocks. For 2x4, we can do:
    #   Split B_block into two 2x2 blocks: B_left (cols 4t, 4t+1) and B_right (cols 4t+2, 4t+3)
    #   But that gives 7+7 = 14 terms per 4-col block, times 4 blocks = 56, too many.
    #
    # Alternative: Use Strassen on the full 2x4 by treating it as 2x(2+2).
    # Actually, the standard approach: for 2xN matrix multiplication, we can use
    # a block Strassen-like approach. But let me think differently.
    #
    # Actually, for the top 2 rows of A (2x3) times B (3x8), we can decompose:
    #   C_top = A_top * B
    #   where A_top is 2x3, B is 3x8.
    #
    # Split A_top into [A_2x2 | A_2x1] and B into [B_top; B_bot] where B_top is 2x8, B_bot is 1x8.
    #   C_top = A_2x2 * B_top + A_2x1 * B_bot
    #
    # For A_2x2 * B_top (2x2 * 2x8):
    #   Use Strassen on each 2x2 column block of B_top. B_top has 4 column blocks of 2x2.
    #   Each 2x2*2x2 uses 7 terms. Total: 4*7 = 28 terms.
    #
    # For A_2x1 * B_bot (2x1 * 1x8):
    #   This is just outer product: 2*8 = 16 terms (each A entry times each B entry).
    #   Actually, it's rank 2*8 = 16 since it's a 2x1 times 1x8.
    #   Wait: (A_2x1 * B_bot)_{i,k} = A_2x1[i] * B_bot[k]
    #   This is a rank-1 outer product for each pair? No, it's:
    #   C[i,k] = A[i] * B[k] for i in 0,1 and k in 0..7
    #   This is a rank-2 tensor? No, it's actually rank 1 per (i,k) pair but the sum is:
    #   C = A_vec ⊗ B_vec, which is rank 1 in the tensor sense? 
    #   No: C[i,k] = A[i]*B[k] is an outer product, rank 1.
    #   But wait, we need it to add to the other term. The term A_2x1 * B_bot contributes
    #   to C_top[i,k] = sum_j A_2x2... no wait.
    #
    # Let me re-examine:
    # A is 3x3, B is 3x8.
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [B1; B2; B3] where B1 is 2x8, B2 is 1x8, B3 is 1x8.
    #
    # C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B3]
    #
    # C_top (rows 0,1) = A11*B1 + A12*B2  (2x8)
    # C_bot (row 2) = A21*B1 + A22*B3  (1x8)
    #
    # For A11*B1 (2x2 * 2x8):
    #   Split B1 into 4 blocks of 2x2: B1_0, B1_1, B1_2, B1_3
    #   Each A11*B1_t uses 7 Strassen terms. Total: 28 terms.
    #
    # For A12*B2 (2x1 * 1x8):
    #   C[i,k] += A12[i] * B2[k]
    #   This is 2*8 = 16 terms (each is a product of one A entry and one B entry).
    #   But can we do better? The tensor for 2x1 * 1x8 has rank min(2,1)*8... 
    #   Actually, the matrix multiplication tensor for (2,1,8) has rank 16 (schoolbook).
    #   Since min(2,1)=1, the rank is 1*8 = 8? No.
    #   The rank of the tensor for m x n * n x p is at most m*n*p, and for 2x1*1x8 it's 16.
    #   Actually, the rank is exactly 16 because each output C[i,k] = A[i]*B[k] is a distinct product.
    #   But wait, we can factor: C[i,k] = A[i] * B[k]. The tensor is sum_{i,k} e_i ⊗ e_k ⊗ e_{ik}
    #   which has rank 16. We can't do better than 16 for this sub-problem in isolation.
    #
    # For A21*B1 (1x2 * 2x8):
    #   C[2,k] += sum_j A21[j] * B1[j,k]
    #   This is 2*8 = 16 terms (schoolbook for 1x2 * 2x8).
    #   Can we do better? The rank of (1,2,8) tensor is 16.
    #   Actually, C[2,k] = A21[0]*B1[0,k] + A21[1]*B1[1,k]
    #   Each term A21[j]*B1[j,k] is a product. We need 2*8 = 16 such products.
    #   The rank is 16.
    #
    # For A22*B3 (1x1 * 1x8):
    #   C[2,k] += A22 * B3[k]
    #   This is 8 terms. Rank = 8.
    #
    # Total rank = 28 + 16 + 16 + 8 = 68.
    
    # Term 1: A11 * B1 (2x2 * 2x8) using Strassen on 4 blocks of 2x2
    # A11 entries: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B1 entries: B[0,k]=V[k], B[1,k]=V[8+k] for k=0..7
    # For block t (t=0..3), B1_t is 2x2 with entries:
    #   B1_t[0,0] = V[4*t], B1_t[0,1] = V[4*t+1]
    #   B1_t[1,0] = V[8+4*t], B1_t[1,1] = V[8+4*t+1]
    # C entries for block t:
    #   C[0,4*t] = W[4*t*3+0], C[0,4*t+1] = W[(4*t+1)*3+0]
    #   C[1,4*t] = W[4*t*3+1], C[1,4*t+1] = W[(4*t+1)*3+1]
    
    for t in range(4):
        b00_row = 4 * t          # V row for B[0, 4t]
        b01_row = 4 * t + 1      # V row for B[0, 4t+1]
        b10_row = 8 + 4 * t      # V row for B[1, 4t]
        b11_row = 8 + 4 * t + 1  # V row for B[1, 4t+1]
        
        c00_row = (4 * t) * 3 + 0      # W row for C[0, 4t]
        c01_row = (4 * t + 1) * 3 + 0  # W row for C[0, 4t+1]
        c10_row = (4 * t) * 3 + 1      # W row for C[1, 4t]
        c11_row = (4 * t + 1) * 3 + 1  # W row for C[1, 4t+1]
        
        # Strassen 7 terms:
        # M1 = (A00+A11)*(B00+B11)
        U[0][col] += 1; U[4][col] += 1
        V[b00_row][col] += 1; V[b11_row][col] += 1
        W[c00_row][col] += 1; W[c11_row][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[3][col] += 1; U[4][col] += 1
        V[b00_row][col] += 1
        W[c10_row][col] += 1; W[c11_row][col] += -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[0][col] += 1
        V[b01_row][col] += 1; V[b11_row][col] += -1
        W[c01_row][col] += 1; W[c10_row][col] += 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[4][col] += 1
        V[b10_row][col] += 1; V[b00_row][col] += -1
        W[c00_row][col] += -1; W[c01_row][col] += 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[0][col] += 1; U[1][col] += 1
        V[b11_row][col] += 1
        W[c00_row][col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[3][col] += 1; U[0][col] += -1
        V[b00_row][col] += 1; V[b01_row][col] += 1
        W[c11_row][col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1][col] += 1; U[4][col] += -1
        V[b10_row][col] += 1; V[b11_row][col] += 1
        W[c00_row][col] += 1; W[c11_row][col] += 1
        col += 1
    
    # Term 2: A12 * B2 (2x1 * 1x8)
    # A12 entries: A[0,2]=U[2], A[1,2]=U[5]
    # B2 entries: B[2,k]=V[16+k] for k=0..7
    # C[0,k] += A[0,2]*B[2,k] => W[k*3+0]
    # C[1,k] += A[1,2]*B[2,k] => W[k*3+1]
    for k in range(8):
        # A[0,2] * B[2,k]
        U[2][col] += 1
        V[16 + k][col] += 1
        W[k * 3 + 0][col] += 1
        col += 1
        # A[1,2] * B[2,k]
        U[5][col] += 1
        V[16 + k][col] += 1
        W[k * 3 + 1][col] += 1
        col += 1
    
    # Term 3: A21 * B1 (1x2 * 2x8)
    # A21 entries: A[2,0]=U[6], A[2,1]=U[7]
    # B1 entries: B[0,k]=V[k], B[1,k]=V[8+k]
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k]
    for k in range(8):
        # A[2,0] * B[0,k]
        U[6][col] += 1
        V[k][col] += 1
        W[k * 3 + 2][col] += 1
        col += 1
        # A[2,1] * B[1,k]
        U[7][col] += 1
        V[8 + k][col] += 1
        W[k * 3 + 2][col] += 1
        col += 1
    
    # Term 4: A22 * B3 (1x1 * 1x8)
    # A22 entry: A[2,2]=U[8]
    # B3 entries: B[2,k]=V[16+k]
    # C[2,k] += A[2,2]*B[2,k]
    for k in range(8):
        U[8][col] += 1
        V[16 + k][col] += 1
        W[k * 3 + 2][col] += 1
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
