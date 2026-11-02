import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # The matrix multiplication tensor T[i, j, k] = 1 if i, j, k are valid indices
    # such that C[i, k] = sum_j A[i, j] B[j, k]
    # Specifically, for each term in the sum, we have a component A[i, j] * B[j, k] contributing to C[i, k]
    # So the tensor has ones at positions (i, j, k) for all valid i, j, k.
    # This is a "completely positive" structure in the sense that the standard decomposition
    # uses rank n*m*p = 2*4*8 = 64.
    
    # However, we can do better. The current solution uses rank 56 by using Strassen's algorithm
    # on 2x2 blocks. Let's think about whether we can improve further.
    
    # The naive rank is 64. Strassen-based gives 56. Can we get lower?
    
    # Actually, let's reconsider. The standard decomposition for multiplying an n x m matrix
    # by an m x p matrix has rank n*m*p when done naively (one rank-1 term for each (i,j,k) triple).
    # For n=2, m=4, p=8, that's 64.
    #
    # The Strassen approach in the current code gives 56. This is because for each pair of 
    # 2x2 blocks (there are 2*4 = 8 such pairs when considering the middle dimension split into
    # two halves of size 2), we use 7 rank-1 terms instead of 4*2 = 8.
    # So 8 * 7 = 56.
    
    # Can we do better? Let's think about the structure more carefully.
    #
    # A is 2x4, B is 4x8, C is 2x8.
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    #
    # One approach: view A as two 2x2 blocks A0, A1 (columns 0-1, 2-3)
    # and B as four 2x8 blocks B0, B1, B2, B3 (rows 0, 1, 2, 3)
    # Then C = A0*B0 + A0*B1 + A1*B2 + A1*B3 where each product is 2x8.
    # Wait, that's not quite right. Let me be more careful.
    
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each product A_left * B_top is 2x8, and A_right * B_bottom is 2x8.
    # 
    # For a 2x2 times 2x8 matrix multiplication, we can use Strassen-like methods.
    # Actually, 2x2 * 2x8 = (2x2 * 2x2) for each pair of columns of B.
    # There are 4 such pairs (columns 0-1, 2-3, 4-5, 6-7).
    # Each 2x2 * 2x2 uses rank 7 (Strassen).
    # So for A_left * B_top: 4 * 7 = 28
    # For A_right * B_bottom: 4 * 7 = 28
    # Total: 56. Same as before.
    
    # Can we reduce further? The question is whether there's a lower rank decomposition.
    # The rank of the matrix multiplication tensor MM_{n,m,p} is known to be at most n*m*p 
    # and at least max(n*p, m*p, n*m) in some cases.
    
    # For our case: n=2, m=4, p=8.
    # Lower bound: max(2*8, 4*8, 2*4) = max(16, 32, 8) = 32.
    # So the rank is at least 32.
    
    # Can we achieve rank 32? That would be ideal. But it's not clear if it's possible.
    # 
    # Actually, I recall that for matrix multiplication, the rank is conjectured to be 
    # close to the "border rank" which for n x m * m x p is approximately (n+m+p)^2 / 4 
    # or similar, but the exact rank is hard to determine.
    
    # Let's try a different approach. Maybe we can use a recursive or more clever decomposition.
    #
    # Alternative: Use the fact that we can group terms differently.
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    #
    # If we could find a decomposition with rank 48 or 40, that would be better.
    #
    # Let's try: split j into two groups: {0,1} and {2,3}.
    # C = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    # = A_sub1 * B_sub1 + A_sub2 * B_sub2
    # where A_sub1 is 2x2, B_sub1 is 2x8, A_sub2 is 2x2, B_sub2 is 2x8.
    #
    # For 2x2 * 2x8, the rank is 2*2*8 / something? 
    # Actually, 2x2 * 2x8: each of the 2 output columns of the 2x2 result requires 
    # computing 2x2 * 2x1 vectors. No wait, B is 2x8, so we're computing 8 independent 
    # 2x1 results, each being 2x2 * 2x1.
    # A 2x2 * 2x1 multiplication has rank... well, it's just 4 scalar multiplications, 
    # so rank 4 for each, but they share the A part.
    #
    # Actually, for fixed A (2x2), computing A * B where B is 2x8 is just 8 independent 
    # vector multiplications, each requiring 4 products. So total 32 products, rank 32 
    # for this sub-problem if done naively. But we can use Strassen on 2x2 blocks.
    #
    # B is 2x8, so we can view it as 4 blocks of 2x2. Each 2x2 * 2x2 uses rank 7.
    # So 2x2 * 2x8 = 4 * 7 = 28.
    # Two such sub-problems: 2 * 28 = 56. Same as before.
    
    # I think 56 might be close to optimal for this approach. Let's stick with 56 
    # but verify the implementation is correct and perhaps optimize the code.
    
    # Let me just keep the existing approach but clean it up.
    
    # Strassen components for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0
                a_map[1] = 0*4 + 1
                a_map[2] = 1*4 + 0
                a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            b_map = [0]*4
            if s == 0:
                b_map[0] = 0*8 + (2*j)
                b_map[1] = 0*8 + (2*j+1)
                b_map[2] = 1*8 + (2*j)
                b_map[3] = 1*8 + (2*j+1)
            else:
                b_map[0] = 2*8 + (2*j)
                b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j)
                b_map[3] = 3*8 + (2*j+1)
                
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc], r] += val
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
