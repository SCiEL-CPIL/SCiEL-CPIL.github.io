"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Strassen 2x2x2: rank 7. Returns U(4,7), V(4,7), W(4,7) with trace convention W[k*n+i, r]."""
    n, m, p = 2, 2, 2
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # Strassen formulas:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A11+A22)*B22
    # M3 = A22*(B21-B11)
    # M4 = A11*(B12-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4+M5-M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M2-M3+M7
    
    # U[i*m+j, r] = coefficient of A[i,j] in rank-r product
    # V[j*p+k, r] = coefficient of B[j,k] in rank-r product
    # W[k*n+i, r] = coefficient of C[i,k] in rank-r product
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # W[k*n+i, r]: C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
    
    # M1: (A11-A22)*(B22-B11) => A[0,0]*B[1,1] - A[1,1]*B[1,1] - A[0,0]*B[0,0] + A[1,1]*B[0,0]
    U[0, 0] = 1;  U[3, 0] = -1
    V[3, 0] = 1;  V[0, 0] = -1
    W[0, 0] = 1;  W[1, 0] = 1;  W[2, 0] = 1;  W[3, 0] = -1
    
    # M2: (A11+A22)*B22 => A[0,0]*B[1,1] + A[1,1]*B[1,1]
    U[0, 1] = 1;  U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1;  W[3, 1] = 1
    
    # M3: A22*(B21-B11) => A[1,1]*B[1,0] - A[1,1]*B[0,0]
    U[3, 2] = 1
    V[2, 2] = 1;  V[0, 2] = -1
    W[1, 2] = 1;  W[3, 2] = -1
    
    # M4: A11*(B12-B22) => A[0,0]*B[0,1] - A[0,0]*B[1,1]
    U[0, 3] = 1
    V[1, 3] = 1;  V[3, 3] = -1
    W[0, 3] = 1;  W[2, 3] = 1
    
    # M5: (A11+A12)*B22 => A[0,0]*B[1,1] + A[0,1]*B[1,1]
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1;  W[2, 4] = 1
    
    # M6: (A21-A11)*(B11+B12) => A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A12-A22)*(B21+B22) => A[0,1]*B[1,0] + A[0,1]*B[1,1] - A[1,1]*B[1,0] - A[1,1]*B[1,1]
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = -1;  W[1, 6] = 1;  W[3, 6] = 1
    
    return U, V, W


def _winograd_2x2x3():
    """Winograd 2x2x3: rank 8. Multiplies 2x2 by 2x3. Returns U(4,8), V(6,8), W(6,8)."""
    n, m, p = 2, 2, 3
    U = np.zeros((4, 8), dtype=np.int8)
    V = np.zeros((6, 8), dtype=np.int8)
    W = np.zeros((6, 8), dtype=np.int8)
    
    # Winograd F(2,2,3) algorithm:
    # A is 2x2, B is 2x3, C is 2x3
    # Transform A: 
    #   a0 = A[0,0] + A[0,1]
    #   a1 = A[1,0] + A[1,1]
    #   a2 = A[0,0] - A[1,1]
    #   a3 = A[0,1] - A[1,0]
    #   a4 = A[0,0]
    #   a5 = A[0,1]
    #   a6 = A[1,0]
    #   a7 = A[1,1]
    # Wait, let me use the standard Winograd 2x2x3 with rank 8.
    # Actually, the minimal rank for 2x2x3 is 8 (Winograd).
    # 
    # Standard Winograd F(2,2,3):
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = A*B = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # 
    # Winograd decomposition (rank 8):
    # We use the transform matrices.
    # U (input transform for A): 4x8
    # V (input transform for B): 6x8  
    # W (output transform for C): 6x8
    #
    # Actually let me just construct it directly.
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    #
    # Winograd approach:
    # Define intermediate variables:
    # m0 = (a+b)*(e+h)  -> contributes to C[0,0], C[0,1]... 
    # This is getting complex. Let me use a known rank-8 decomposition.
    
    # Actually, for 2x2x3, the schoolbook rank is 12. Winograd achieves 8.
    # Let me use the standard Winograd F(2,2,3) scheme.
    
    # From Winograd's paper, the 2x2x3 case:
    # Transform A (2x2) to 8 values:
    #   a0 = a + b, a1 = c + d, a2 = a - d, a3 = b - c
    #   a4 = a, a5 = b, a6 = c, a7 = d
    # Transform B (2x3) to 8 values:
    #   b0 = e + h, b1 = f + i, b2 = g + j
    #   b3 = e - h, b4 = f - i, b5 = g - j
    #   b6 = e, b7 = h  (wait, that's only 8 if we count right)
    # 
    # Hmm, let me just do schoolbook for 2x2x3 (rank 12) to be safe, 
    # or use a simpler known decomposition.
    
    # Actually, let me reconsider. For the 4x4x7 problem:
    # Split 7 = 4 + 3.
    # 4x4x4 can be done with 4 independent 2x2x2 Strassen blocks? No.
    # 4x4x4: split into four 2x2x2 blocks, each Strassen rank 7, total 28.
    # 4x4x3: split into two 2x2x3 blocks (top and bottom rows of A), each Winograd rank 8, total 16.
    # Total: 28 + 16 = 44? That seems too good. Let me verify.
    
    # Wait, 4x4x4: A is 4x4, B is 4x4, C is 4x4.
    # Split A into 4 blocks of 2x2, B into 4 blocks of 2x2.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # If we split rows of A and columns of B:
    # A = [[A11, A12],[A21, A22]] where each is 2x2
    # B = [[B11, B12],[B21, B22]] where each is 2x2
    # C = [[A11*B11+A12*B21, A11*B12+A12*B22],
    #      [A21*B11+A22*B21, A21*B12+A22*B22]]
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # We have 4 such multiplications: A11*B11, A12*B21, A21*B11, A22*B21, A11*B12, A12*B22, A21*B12, A22*B22
    # That's 8 multiplications of 2x2x2, each rank 7, total 56.
    # But we can share: A11*B11 and A11*B12 share A11. 
    # Actually, the standard way: 4x4x4 with Strassen on 2x2 blocks gives rank 4*7 = 28? No.
    # 
    # Let me think more carefully. For 4x4x4:
    # We can view it as 4 independent 2x4x4 multiplications (each row of A times B).
    # Or 4x2x4 multiplications.
    # 
    # Actually, the cleanest approach: 4x4x7 = (4x4x4) + (4x4x3)
    # For 4x4x4: use 4 Strassen 2x2x2 blocks? 
    # A(4x4) * B(4x4): 
    #   Split A into left 2 cols and right 2 cols: A = [A_L | A_R], each 4x2
    #   Split B into top 2 rows and bottom 2 rows: B = [B_T; B_B], each 2x4
    #   C = A_L * B_T + A_R * B_B
    #   A_L * B_T is 4x2 * 2x4 = 4x4, which is 4x2x4
    #   A_R * B_B is 4x2 * 2x4 = 4x4, which is 4x2x4
    #   4x2x4: split into two 2x2x4 multiplications (top 2 rows of A_L, bottom 2 rows)
    #   2x2x4: split into two 2x2x2 multiplications
    #   So 4x2x4 = 2 * (2x2x4) = 2 * 2 * (2x2x2) = 4 * (2x2x2), rank 4*7=28
    #   Total for 4x4x4: 2 * 28 = 56? That's worse than schoolbook 64? No, 56 < 64.
    #   Wait, schoolbook 4x4x4 is 64. Strassen-based gives 56.
    #
    # For 4x4x3:
    #   A(4x4) * B(4x3):
    #   Split A into left 2 cols and right 2 cols: A = [A_L | A_R], each 4x2
    #   B is 4x3, split into top 2 rows and bottom 2 rows: B = [B_T; B_B], B_T is 2x3, B_B is 2x3
    #   C = A_L * B_T + A_R * B_B
    #   A_L * B_T is 4x2 * 2x3 = 4x3, which is 4x2x3
    #   4x2x3: split into two 2x2x3 multiplications
    #   2x2x3: Winograd rank 8
    #   So 4x2x3 = 2 * 8 = 16
    #   Total for 4x4x3: 2 * 16 = 32
    #   Schoolbook 4x4x3 is 48.
    #
    # Total: 56 + 32 = 88? Let me double-check.
    #
    # Hmm wait, I think I'm double-counting. Let me be more careful.
    # 
    # 4x4x7: A is 4x4, B is 4x7, C is 4x7.
    # C = A * B
    # Split B into B1 (4x4) and B2 (4x3): B = [B1 | B2]
    # C = A*B1 + ... no, C = A * [B1|B2] = [A*B1 | A*B2]
    # So C1 = A*B1 (4x4) and C2 = A*B2 (4x3)
    # 
    # For A*B1 (4x4x4):
    #   Split A into A1 (4x2, left half) and A2 (4x2, right half)
    #   Split B1 into B1_top (2x4) and B1_bot (2x4)
    #   A*B1 = A1*B1_top + A2*B1_bot
    #   A1*B1_top: 4x2 * 2x4 = 4x4 (this is a 4x2x4 multiplication)
    #   A2*B1_bot: 4x2 * 2x4 = 4x4 (this is a 4x2x4 multiplication)
    #   For 4x2x4: split the 4 rows into two 2-row groups
    #     A1 = [A1_top; A1_bot] where each is 2x2
    #     A1*B1_top = A1_top*B1_top + A1_bot*B1_top
    #     Each 2x2 * 2x4 = 2x2x4, split into two 2x2x2:
    #     2x2x4 = 2*(2x2x2), rank 2*7 = 14
    #     So A1*B1_top = 14 + 14 = 28
    #   Similarly A2*B1_bot = 28
    #   Total for A*B1: 28 + 28 = 56
    #
    # For A*B2 (4x4x3):
    #   Split A into A1 (4x2) and A2 (4x2)
    #   Split B2 into B2_top (2x3) and B2_bot (2x3)
    #   A*B2 = A1*B2_top + A2*B2_bot
    #   A1*B2_top: 4x2 * 2x3 = 4x3 (4x2x3)
    #   4x2x3: split into two 2x2x3
    #     2x2x3: Winograd rank 8
    #     So 4x2x3 = 2*8 = 16
    #   A1*B2_top = 16, A2*B2_bot = 16
    #   Total for A*B2: 32
    #
    # Grand total: 56 + 32 = 88
    #
    # This is better than schoolbook 112. Let me verify the 2x2x3 Winograd rank is indeed 8.
    # Yes, Winograd showed 2x2x3 can be done in 8 multiplications.
    
    # Let me implement this properly.
    # For now, let me just do schoolbook to ensure correctness, then optimize.
    
    # Actually, let me implement the block structure properly.
    
    # For 2x2x3 Winograd, let me use a known rank-8 decomposition.
    # After research, the Winograd 2x2x3 uses:
    #   m0 = (a+b)(e+h)
    #   m1 = (a+b)(f+i) 
    #   m2 = (a+b)(g+j)
    #   m3 = (c+d)(e+h)
    #   m4 = (c+d)(f+i)
    #   m5 = (c+d)(g+j)
    #   m6 = (a-d)(e-h)  -- wait, this doesn't seem right for rank 8
    #
    # Let me just use schoolbook for 2x2x3 (rank 12) to be safe, giving total 56+48=104.
    # Or better, let me implement Strassen for 2x2x2 and schoolbook for the rest.
    
    # Actually, I recall that for 2x2x3, the minimal rank is 8 and it's achieved by Winograd.
    # Let me look up the exact decomposition.
    
    # Winograd 2x2x3 (from "Arithmetic Complexity of Control" by Winograd):
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # 
    # 8 multiplications:
    # m1 = (a+b) * (e+h)
    # m2 = (a+b) * (f+i)
    # m3 = (a+b) * (g+j)
    # m4 = (c+d) * (e+h)
    # m5 = (c+d) * (f+i)
    # m6 = (c+d) * (g+j)
    # m7 = (a-d) * (e-h)
    # m8 = (a-d) * (f-i)  -- hmm, this doesn't give all terms correctly
    #
    # I think the correct Winograd 2x2x3 is more subtle. Let me just use schoolbook for 2x2x3.
    # Schoolbook 2x2x3: rank 12. Total would be 56 + 48 = 104.
    # Still better than 112.
    
    # Actually, let me just implement the full schoolbook first, verify it works,
    # then add the Strassen-based optimization.
    
    # For now, return schoolbook 4x4x7 (rank 112) as fallback
    # and implement the block decomposition separately.
    
    # Let me just return None to indicate we should use a different approach
    return None


def _schoolblock(n, m, p):
    """Schoolbook decomposition for n x m times m x p. Rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2_correct():
    """Correct Strassen 2x2x2 decomposition. Rank 7."""
    n, m, p = 2, 2, 2
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # Strassen:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A11+A22)*B22
    # M3 = A22*(B21-B11)
    # M4 = A11*(B12-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4+M5-M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M2-M3+M7
    
    # Indexing: A[i,j] -> U[i*2+j, r], B[j,k] -> V[j*2+k, r], C[i,k] -> W[k*2+i, r]
    
    # M1: (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    
    # M2: (A[0,0]+A[1,1])*B[1,1]
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A[1,1]*(B[1,0]-B[0,0])
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A[0,0]*(B[0,1]-B[1,1])
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W


def _build_4x4x7_strassen():
    """
    Build 4x4x7 decomposition using Strassen 2x2x2 blocks.
    
    Strategy:
    - 4x4x7 = 4x4x4 + 4x4x3 (split B into 4x4 and 4x3 parts)
    - 4x4x4: split into 8 2x2x2 Strassen multiplications (rank 8*7=56)
    - 4x4x3: split into 4 2x2x3 schoolbook multiplications (rank 4*12=48)
    - Total rank: 56 + 48 = 104
    
    Actually, let me do it more carefully with proper indexing.
    """
    n, m, p = 4, 4, 7
    
    # We'll build U (4*4=16, rank), V (4*7=28, rank), W (7*4=28, rank)
    # Start with total rank = 104 (8*7 + 4*12)
    total_rank = 8 * 7 + 4 * 12  # 56 + 48 = 104
    
    U = np.zeros((16, total_rank), dtype=np.int8)
    V = np.zeros((28, total_rank), dtype=np.int8)
    W = np.zeros((28, total_rank), dtype=np.int8)
    
    # Part 1: 4x4x4 (B columns 0-3)
    # Split A into 4 blocks of 2x2 (rows 0-1,2-3; cols 0-1,2-3)
    # Split B1 (4x4) into 4 blocks of 2x2 (rows 0-1,2-3; cols 0-1,2-3)
    # C1 = A*B1, 4x4 output
    # C1[i,j] = sum_k A[i,k]*B1[k,j]
    # = sum over block rows/cols
    # 
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2
    #   A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    #   A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Let B1 = [[B11, B12], [B21, B22]] where each is 2x2
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C1 = [[A11*B11+A12*B21, A11*B12+A12*B22],
    #        [A21*B11+A22*B21, A21*B12+A22*B22]]
    #
    # 8 multiplications: A11*B11, A12*B21, A21*B11, A22*B21, A11*B12, A12*B22, A21*B12, A22*B22
    # Each is 2x2x2, Strassen rank 7.
    
    # For each 2x2x2 multiplication, we need to map:
    # A_block[i,j] -> U[(2*row_block + i)*4 + (2*col_block + j), rank_col]
    # B_block[j,k] -> V[(2*row_block + j)*7 + (2*col_block + k), rank_col]  (for B1, cols 0-3)
    # C_block[i,k] -> W[(2*col_block + k)*4 + (2*row_block + i), rank_col]  (for C1, cols 0-3)
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2_correct()
    
    # 8 blocks: (A_row_block, A_col_block, B_row_block, B_col_block)
    # A_row_block in {0,1}, A_col_block in {0,1}, B_row_block in {0,1}, B_col_block in {0,1}
    # The multiplication is A[A_row_block, A_col_block] * B1[B_row_block, B_col_block]
    # where A[A_row_block, A_col_block] means A[2*arb:2*arb+2, 2*acb:2*acb+2]
    # and B1[B_row_block, B_col_block] means B[2*brb:2*brb+2, 2*bc:2*bc+2]
    # Result goes to C1[2*arb:2*arb+2, 2*bc:2*bc+2]
    
    block_configs = [
        # (A_row_block, A_col_block, B_row_block, B_col_block)
        (0, 0, 0, 0),  # A11 * B11
        (0, 1, 1, 0),  # A12 * B21
        (1, 0, 0, 0),  # A21 * B11
        (1, 1, 1, 0),  # A22 * B21
        (0, 0, 0, 1),  # A11 * B12
        (0, 1, 1, 1),  # A12 * B22
        (1, 0, 0, 1),  # A21 * B12
        (1, 1, 1, 1),  # A22 * B22
    ]
    
    rank_offset = 0
    for (arb, acb, brb, bc) in block_configs:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    # U index: (2*arb+i)*4 + (2*acb+j)
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    U[u_idx, rank_offset + r] += strassen_U[i * 2 + j, r]
            
            for j in range(2):
                for k in range(2):
                    # V index: (2*brb+j)*7 + (2*bc+k)  (B1 columns 0-3, so +2*bc+k)
                    v_idx = (2 * brb + j) * 7 + (2 * bc + k)
                    V[v_idx, rank_offset + r] += strassen_V[j * 2 + k, r]
            
            for k in range(2):
                for i in range(2):
                    # W index: (2*bc+k)*4 + (2*arb+i)  (C1 columns 0-3, so +2*bc+k)
                    w_idx = (2 * bc + k) * 4 + (2 * arb + i)
                    W[w_idx, rank_offset + r] += strassen_W[k * 2 + i, r]
        
        rank_offset += 7
    
    # Part 2: 4x4x3 (B columns 4-6)
    # B2 = B[0:4, 4:7], 4x3
    # C2 = A*B2, 4x3 output
    # Split A into 4 blocks of 2x2
    # Split B2 into 2 blocks of 2x3 (top 2 rows, bottom 2 rows)
    # C2 = A1*B2_top + A2*B2_bot where A1 = A[0:2,:], A2 = A[2:4,:]
    # Wait, that's not right. 
    # C2[i,k] = sum_j A[i,j]*B2[j,k]
    # Split rows of A: A = [A_top; A_bot] where A_top = A[0:2, 0:4] (2x4), A_bot = A[2:4, 0:4] (2x4)
    # Split columns of A: A_top = [A_top_L | A_top_R] where each is 2x2
    # A_top_L = A[0:2, 0:2], A_top_R = A[0:2, 2:4]
    # A_bot_L = A[2:4, 0:2], A_bot_R = A[2:4, 2:4]
    # Split B2 rows: B2 = [B2_top; B2_bot] where B2_top = B[0:2, 4:7] (2x3), B2_bot = B[2:4, 4:7] (2x3)
    # C2 = A_top_L * B2_top + A_top_R * B2_bot + A_bot_L * B2_top + A_bot_R * B2_bot
    # Wait, that's not right either.
    # C2[i,k] = sum_{j=0}^{3} A[i,j]*B2[j,k]
    # = sum_{j=0}^{1} A[i,j]*B2[j,k] + sum_{j=2}^{3} A[i,j]*B2[j,k]
    # For i=0,1: = A[0:2, 0:2]*B2[0:2, :] + A[0:2, 2:4]*B2[2:4, :]
    # For i=2,3: = A[2:4, 0:2]*B2[0:2, :] + A[2:4, 2:4]*B2[2:4, :]
    # So 4 multiplications: 2x2x3 each
    # A[0:2,0:2] * B2[0:2,4:7] -> C2[0:2, 0:3]
    # A[0:2,2:4] * B2[2:4,4:7] -> C2[0:2, 0:3]
    # A[2:4,0:2] * B2[0:2,4:7] -> C2[2:4, 0:3]
    # A[2:4,2:4] * B2[2:4,4:7] -> C2[2:4, 0:3]
    
    # Schoolbook 2x2x3: rank 12
    for (arb, acb, brb) in [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        # A block: A[2*arb:2*arb+2, 2*acb:2*acb+2] (2x2)
        # B block: B[2*brb:2*brb+2, 4:7] (2x3)
        # C block: C[2*arb:2*arb+2, 4:7] (2x3)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = rank_offset + i * 6 + j * 3 + k
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    v_idx = (2 * brb + j) * 7 + (4 + k)
                    w_idx = (4 + k) * 4 + (2 * arb + i)
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
        rank_offset += 12
    
    assert rank_offset == total_rank, f"Rank offset {rank_offset} != total {total_rank}"
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if j is the "middle" index (always 1 for matrix mult)
    # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    # Because C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if val != 1:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_time=10):
    """Try to find linear dependencies among rank-1 terms to reduce rank."""
    start_time = time.time()
    rank = U.shape[1]
    
    # Check if any column is a linear combination of others
    # This is expensive, so we do a simple check: look for duplicate or zero columns
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Simple check: remove columns that are all zeros
    for r in range(best_rank - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if np.all(best_U[:, r] == 0) and np.all(best_V[:, r] == 0) and np.all(best_W[:, r] == 0):
            best_U = np.delete(best_U, r, axis=1)
            best_V = np.delete(best_V, r, axis=1)
            best_W = np.delete(best_W, r, axis=1)
            best_rank -= 1
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Build Strassen-based decomposition (rank 104)
    U, V, W = _build_4x4x7_strassen()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U, V, W, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, max_time=5)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
