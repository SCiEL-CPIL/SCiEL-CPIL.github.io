import time
import numpy as np

def _strassen_2x2_2x2():
    # Returns U (4,7), V (4,7), W (4,7) for 2x2 matrix multiplication
    # Using the standard Strassen decomposition
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00 + A11
    V[0,0] = 1; V[3,0] = 1  # B00 + B11
    W[0,0] = 1; W[3,0] = 1  # C00 += M1, C11 += M1
    
    # Term 2: M2 = (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1  # A10 + A11
    V[0,1] = 1              # B00
    W[2,1] = 1; W[3,1] = -1 # C10 += M2, C11 -= M2
    
    # Term 3: M3 = A00*(B10-B11)
    U[0,2] = 1              # A00
    V[2,2] = 1; V[3,2] = -1 # B10 - B11
    W[1,2] = 1; W[3,2] = 1  # C01 += M3, C11 += M3
    
    # Term 4: M4 = A11*(B01-B10)
    U[3,3] = 1              # A11
    V[1,3] = 1; V[2,3] = -1 # B01 - B10
    W[0,3] = 1; W[2,3] = 1  # C00 += M4, C10 += M4
    
    # Term 5: M5 = (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1  # A00 + A01
    V[3,4] = 1              # B11
    W[0,4] = -1; W[1,4] = 1 # C00 -= M5, C01 += M5
    
    # Term 6: M6 = (A10-A00)*(B00+B10)
    U[2,5] = 1; U[0,5] = -1 # A10 - A00
    V[0,5] = 1; V[2,5] = 1  # B00 + B10
    W[3,5] = 1              # C11 += M6
    
    # Term 7: M7 = (A01-A11)*(B01+B11)
    U[1,6] = 1; U[3,6] = -1 # A01 - A11
    V[1,6] = 1; V[3,6] = 1  # B01 + B11
    W[0,6] = 1              # C00 += M7
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = 4, 4, 7
    # Split the 4x7 output into a 4x4 part and a 4x3 part
    # For the 4x4 part, use Strassen-based decomposition (rank 49)
    # For the 4x3 part, use trivial decomposition (rank 4*4*3 = 48)
    # Total rank = 49 + 48 = 97
    
    # --- Part 1: 4x4x4 using Strassen (rank 49) ---
    # We can use 4x4 = 2x2 block matrix multiplication where each block is 2x2
    # The 4x4 matrix multiplication can be decomposed using Strassen on 2x2 blocks
    # This gives rank 7 * (2*2) = 7 * 4 = 28? No.
    # Actually, for 4x4 matrix multiplication using block Strassen:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # We apply Strassen to get 7 products of 2x2 matrices
    # Each 2x2 product has rank 7 (from Strassen) or 4 (trivial)
    # Using Strassen for each 2x2 sub-product: 7 * 7 = 49
    # Using trivial for each 2x2 sub-product: 7 * 4 = 28
    
    # Let's use the trivial 2x2 for each of the 7 Strassen terms: rank = 7*4 = 28
    # Or use Strassen for each 2x2: rank = 7*7 = 49
    # 28 is better, so use trivial 2x2 for each Strassen term
    
    # Build the 4x4x4 decomposition with rank 28
    # Strassen terms for 4x4 blocks (each term is a 2x2 multiplication)
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # For the 4x4x4 part, we use block structure
    # A is 4x4, B is 4x4, C is 4x4
    # Block indices: A00=[0:2,0:2], A01=[0:2,2:4], A10=[2:4,0:2], A11=[2:4,2:4]
    # Similarly for B and C
    # Strassen coefficients map block-level to 2x2-level
    
    # The 7 Strassen terms each involve a 2x2 matrix multiplication
    # For term t (0..6), we have 2x2 matrices (U_t, V_t, W_t) from _strassen_2x2_2x2
    # But actually, we need to think about this differently.
    
    # Let's just use the trivial decomposition for the 4x4x4 part too, 
    # but that gives rank 64. With Strassen at the block level, we get 7*4=28.
    
    # Actually, let me reconsider. The simplest approach:
    # Use Strassen for 4x4x4: decompose into 7 products of 2x2 matrices
    # Each 2x2 product uses trivial decomposition (rank 4)
    # Total rank for 4x4x4 part = 7 * 4 = 28
    
    # Build this decomposition:
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    # Strassen block combinations
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # For each term, we need to form the 2x2 input matrices
    # Then multiply them using trivial 2x2 decomposition
    
    # Let's define the block structure more carefully
    # A is 4x4: rows 0-3, cols 0-3
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # B is 4x4: rows 0-3, cols 0-3
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # C is 4x4: rows 0-3, cols 0-3
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4]
    # C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # For each Strassen term, we compute a 2x2 product
    # The 2x2 product C_small = A_small * B_small
    # A_small is 2x2, B_small is 2x2, C_small is 2x2
    
    # Let me just build it directly using the known Strassen coefficients
    
    # U for A blocks: each term contributes to certain A blocks
    # V for B blocks: each term contributes to certain B blocks
    # W for C blocks: each term contributes to certain C blocks
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # A_small = A00 + A11 (2x2), B_small = B00 + B11 (2x2)
    # For trivial 2x2: C_small[i,j] = sum_k A_small[i,k] * B_small[k,j]
    # C_small contributes to C00 and C11
    
    # This is getting complex. Let me just use the trivial decomposition for both parts.
    # 4x4x4: rank 64, 4x4x3: rank 48, total = 112. Same as before.
    
    # Actually, let me try a different approach.
    # Use the known optimal rank for 4x4x4 which is 28 (from Strassen with trivial 2x2 sub-multiplications)
    
    # Build 4x4x4 with rank 28:
    rank_4x4 = 28
    U_4x4 = [[0] * rank_4x4 for _ in range(16)]
    V_4x4 = [[0] * rank_4x4 for _ in range(16)]
    W_4x4 = [[0] * rank_4x4 for _ in range(16)]
    
    # Strassen terms and their block contributions
    # Each term t (0-6) has 4 sub-terms (for the 2x2 trivial multiplication)
    # Total = 7 * 4 = 28
    
    # For term t, sub-term s (0-3):
    # s = i*2 + j for the 2x2 indices? No, for 2x2 trivial:
    # C_small[i,j] = A_small[i,k] * B_small[k,j]
    # Where (i,j,k) ranges over 0,1
    
    # Let me define the Strassen block coefficients
    # For each term t, we have:
    # A_small = sum over blocks a_t * A_block
    # B_small = sum over blocks b_t * B_block
    # C_small contributes to C_blocks with coefficients c_t
    
    # Term 0 (M1): A00+A11, B00+B11, contributes to C00(+1), C11(+1)
    # Term 1 (M2): A10+A11, B00, contributes to C10(+1), C11(-1)
    # Term 2 (M3): A00, B10-B11, contributes to C01(+1), C11(+1)
    # Term 3 (M4): A11, B01-B10, contributes to C00(+1), C10(+1)
    # Term 4 (M5): A00+A01, B11, contributes to C00(-1), C01(+1)
    # Term 5 (M6): A10-A00, B00+B10, contributes to C11(+1)
    # Term 6 (M7): A01-A11, B01+B11, contributes to C00(+1)
    
    # Block to index mapping for 4x4 matrix (row-major, 16 elements)
    # A00: indices 0,1,4,5 (rows 0-1, cols 0-1)
    # A01: indices 2,3,6,7 (rows 0-1, cols 2-3)
    # A10: indices 8,9,12,13 (rows 2-3, cols 0-1)
    # A11: indices 10,11,14,15 (rows 2-3, cols 2-3)
    
    # Similarly for B and C
    
    # For the 2x2 sub-multiplication within each term:
    # A_small is 2x2, B_small is 2x2
    # A_small[i,k] corresponds to A_block element
    # For a given block, the 2x2 elements are at specific positions
    
    # This is very complex. Let me just use the trivial decomposition for everything.
    # Rank = 4*4*4 + 4*4*3 = 64 + 48 = 112.
    
    # Actually, I realize I should just implement a proper block Strassen.
    # Let me define it carefully.
    
    # For 4x4 = 2x2 blocks of 2x2:
    # A blocks: A00 (2x2), A01 (2x2), A10 (2x2), A11 (2x2)
    # In the 4x4 matrix A (flattened row-major):
    # A00: A[0,0], A[0,1], A[1,0], A[1,1] -> flat indices 0,1,4,5
    # A01: A[0,2], A[0,3], A[1,2], A[1,3] -> flat indices 2,3,6,7
    # A10: A[2,0], A[2,1], A[3,0], A[3,1] -> flat indices 8,9,12,13
    # A11: A[2,2], A[2,3], A[3,2], A[3,3] -> flat indices 10,11,14,15
    
    # For each Strassen term t, we form:
    # A_small (2x2) from linear combination of A blocks
    # B_small (2x2) from linear combination of B blocks
    # Then C_small = A_small * B_small (2x2)
    # C_small contributes to C blocks
    
    # The 2x2 multiplication A_small * B_small uses trivial decomposition:
    # C_small[0,0] = A_small[0,0]*B_small[0,0] + A_small[0,1]*B_small[1,0]
    # C_small[0,1] = A_small[0,0]*B_small[0,1] + A_small[0,1]*B_small[1,1]
    # C_small[1,0] = A_small[1,0]*B_small[0,0] + A_small[1,1]*B_small[1,0]
    # C_small[1,1] = A_small[1,0]*B_small[0,1] + A_small[1,1]*B_small[1,1]
    
    # Each of these 4 products is a scalar multiplication, so we need 4 rank-1 terms per Strassen term
    # Total: 7 * 4 = 28 terms
    
    # For term t and sub-term s (0-3):
    # s=0: C_small[0,0] = A_small[0,0]*B_small[0,0] + A_small[0,1]*B_small[1,0]
    # This involves 2 products, so we need 2 rank-1 terms?
    # No, for the trivial 2x2 decomposition, each output element is a sum of 2 products
    # So for C_small[0,0], we have 2 terms: (A[0,0]*B[0,0]) and (A[0,1]*B[1,0])
    
    # Wait, I'm confusing myself. The trivial 2x2 decomposition has rank 4:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Each product A[i,k]*B[k,j] is one rank-1 term in the 2x2 tensor
    # There are 2*2*2 = 8 such products? No, for 2x2x2, rank is 4:
    # The 4 products are:
    # (i=0,k=0,j=0): A[0,0]*B[0,0] -> C[0,0]
    # (i=0,k=1,j=0): A[0,1]*B[1,0] -> C[0,0]
    # (i=0,k=0,j=1): A[0,0]*B[0,1] -> C[0,1]
    # (i=0,k=1,j=1): A[0,1]*B[1,1] -> C[0,1]
    # (i=1,k=0,j=0): A[1,0]*B[0,0] -> C[1,0]
    # (i=1,k=1,j=0): A[1,1]*B[1,0] -> C[1,0]
    # (i=1,k=0,j=1): A[1,0]*B[0,1] -> C[1,1]
    # (i=1,k=1,j=1): A[1,1]*B[1,1] -> C[1,1]
    
    # That's 8 terms, but the rank of 2x2x2 is 4, not 8.
    # The trivial decomposition uses rank 8? No, the standard trivial is:
    # For each (i,j,k), one term: U[i,k]=1, V[k,j]=1, W[i,j]=1
    # That's 2*2*2 = 8 terms.
    
    # But Strassen gives rank 7 for 2x2x2.
    # So for 4x4x4 using block Strassen with Strassen sub-multiplications: 7*7 = 49
    # With trivial sub-multiplications: 7*8 = 56
    
    # Neither is better than the overall trivial 64 for 4x4x4.
    
    # Let me just go with the trivial decomposition for the whole thing.
    # Rank = 4*4*7 = 112.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
