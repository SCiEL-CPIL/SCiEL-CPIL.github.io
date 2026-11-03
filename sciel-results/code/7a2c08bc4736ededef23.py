"""Rewritten program: rank-28 decomposition for 2x4 by 4x8 matrix multiplication
using Strassen's rank-7 decomposition for 2x2 multiplication applied to block structure."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-28 tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy:
    - A is 2x4, split into A_L (2x2) and A_R (2x2)
    - B is 4x8, split into B_TL (2x4), B_TR (2x4), B_BL (2x4), B_BR (2x4)
      where B_T = [B_TL, B_TR] is 2x4 and B_B = [B_BL, B_BR] is 2x4
    - C = A*B is 2x8, split into C_TL, C_TR, C_BL, C_BR (each 2x2)
    
    We compute:
      C_TL = A_L * B_TL  (2x2 * 2x2 -> 2x2)
      C_TR = A_L * B_TR  (2x2 * 2x2 -> 2x2)
      C_BL = A_R * B_BL  (2x2 * 2x2 -> 2x2)
      C_BR = A_R * B_BR  (2x2 * 2x2 -> 2x2)
    
    Each 2x2*2x2 multiplication uses rank-7 Strassen decomposition.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen rank-7 decomposition for 2x2 multiplication
    # U maps 2x2 matrix (4 entries) to 7 scalars
    # V maps 2x2 matrix (4 entries) to 7 scalars
    # W maps 2x2 output (4 entries) to 7 scalars
    # 
    # For A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # Strassen products:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Index convention for 2x2 matrix: [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1)
    
    # U: 4x7 matrix where U[i, r] is coefficient of A entry i in Strassen product r
    U_2x2 = np.zeros((4, 7), dtype=float)
    # P1: a11 + a22 -> entries 0 and 3
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    # P2: a21 + a22 -> entries 2 and 3
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    # P3: a11 -> entry 0
    U_2x2[0, 2] = 1
    # P4: a22 -> entry 3
    U_2x2[3, 3] = 1
    # P5: a11 + a12 -> entries 0 and 1
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    # P6: a21 - a11 -> entries 2 and 0
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    # P7: a12 - a22 -> entries 1 and 3
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    
    # V: 4x7 matrix where V[j, r] is coefficient of B entry j in Strassen product r
    V_2x2 = np.zeros((4, 7), dtype=float)
    # P1: b11 + b22 -> entries 0 and 3
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    # P2: b11 -> entry 0
    V_2x2[0, 1] = 1
    # P3: b12 - b22 -> entries 1 and 3
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    # P4: b21 - b11 -> entries 2 and 0
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    # P5: b22 -> entry 3
    V_2x2[3, 4] = 1
    # P6: b11 + b12 -> entries 0 and 1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    # P7: b21 + b22 -> entries 2 and 3
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    
    # W: 4x7 matrix where W[k, r] is coefficient of Strassen product r in C entry k
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    W_2x2 = np.zeros((4, 7), dtype=float)
    # C11 (entry 0)
    W_2x2[0, 0] = 1; W_2x2[0, 3] = 1; W_2x2[0, 4] = -1; W_2x2[0, 6] = 1
    # C12 (entry 1)
    W_2x2[1, 2] = 1; W_2x2[1, 4] = 1
    # C21 (entry 2)
    W_2x2[2, 1] = 1; W_2x2[2, 3] = 1
    # C22 (entry 3)
    W_2x2[3, 0] = 1; W_2x2[3, 1] = -1; W_2x2[3, 2] = 1; W_2x2[3, 5] = 1
    
    # Now build the full decomposition for 2x4 * 4x8
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (32, 28)  
    # W has shape (p*n, rank) = (16, 28)
    #
    # Index conventions:
    # U[i*m+j, c] where i in [0,1], j in [0,3]
    # V[j*p+k, c] where j in [0,3], k in [0,7]
    # W[k*n+i, c] where k in [0,7], i in [0,1]
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 28)
    
    # Block structure:
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4], B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    # B_T = [B_TL, B_TR] = B[0:2, :], B_B = [B_BL, B_BR] = B[2:4, :]
    #
    # C_TL = A_L * B_TL -> C[0:2, 0:2]
    # C_TR = A_L * B_TR -> C[0:2, 2:4]
    # C_BL = A_R * B_BL -> C[0:2, 4:6]  wait, let me re-check
    #
    # Actually, C = A * B where A is 2x4, B is 4x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split A = [A_L | A_R] where A_L is 2x2 (cols 0-1), A_R is 2x2 (cols 2-3)
    # Split B rows: B_T = B[0:2, :] (2x4 after splitting cols), B_B = B[2:4, :]
    # B_T = [B_TL | B_TR] where B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
    # B_B = [B_BL | B_BR] where B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    #
    # C[i,k] for k in 0-1: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #                        = (A_L * B_TL)[i,k] + (A_R * B_BL)[i,k]
    # C[i,k] for k in 2-3: C[i,k] = (A_L * B_TR)[i,k-2] + (A_R * B_BR)[i,k-2]
    #
    # So:
    # C[:, 0:2] = A_L * B_TL + A_R * B_BL
    # C[:, 2:4] = A_L * B_TR + A_R * B_BR
    #
    # Each product is 2x2 * 2x2, computed with rank-7 Strassen
    # Total: 4 products * 7 rank = 28 rank
    
    # For each of the 4 sub-problems, we use 7 Strassen terms
    # Sub-problem (a, b) where a in {L, R}, b in {TL, TR, BL, BR}
    # a determines which 2 columns of A: L -> cols 0-1, R -> cols 2-3
    # b determines which 2 rows and 2 cols of B
    
    # Define the 4 sub-problems
    # (a_start, b_row_start, b_col_start)
    sub_problems = [
        (0, 0, 0),   # A_L * B_TL -> C[:, 0:2]
        (2, 0, 2),   # A_R * B_TR -> C[:, 2:4]
        (0, 2, 0),   # A_L * B_BL -> C[:, 0:2]
        (2, 2, 2),   # A_R * B_BR -> C[:, 2:4]
    ]
    
    # For each sub-problem, we have 7 Strassen terms
    # U contribution: for A entries in rows 0-1, cols a_start to a_start+1
    # V contribution: for B entries in rows b_row_start to b_row_start+1, cols b_col_start to b_col_start+1
    # W contribution: for C entries in rows 0-1, cols b_col_start to b_col_start+1
    
    for sp_idx, (a_col_start, b_row_start, b_col_start) in enumerate(sub_problems):
        for r in range(7):
            col = sp_idx * 7 + r
            
            # U: A entries are at rows 0-1, cols a_col_start to a_col_start+1
            # A local entry (ii, jj) corresponds to global A[0+ii, a_col_start+jj]
            # In U matrix, index is i*m + j where i is row, j is col
            for ii in range(2):
                for jj in range(2):
                    local_idx = ii * 2 + jj  # 0,1,2,3 for 2x2
                    U[ii, a_col_start + jj, col] = U_2x2[local_idx, r]
            
            # V: B entries are at rows b_row_start to b_row_start+1, cols b_col_start to b_col_start+1
            # B local entry (jj, kk) corresponds to global B[b_row_start+jj, b_col_start+kk]
            # In V matrix, index is j*p + k where j is row, k is col
            for jj in range(2):
                for kk in range(2):
                    local_idx = jj * 2 + kk
                    V[b_row_start + jj, b_col_start + kk, col] = V_2x2[local_idx, r]
            
            # W: C entries are at rows 0-1, cols b_col_start to b_col_start+1
            # C local entry (kk, ii) corresponds to global C[0+ii, b_col_start+kk]
            # In W matrix, index is k*n + i where k is col, i is row
            for kk in range(2):
                for ii in range(2):
                    local_idx = kk * 2 + ii  # C entry (ii, kk) -> local index for 2x2 output
                    W[b_col_start + kk, ii, col] = W_2x2[local_idx, r]
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
