"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strategy: Use Kronecker product structure with 2x2 rank-7 decomposition
    # For 3x4 * 4x4:
    # Split rows of A: top 2 rows (i=0,1) and bottom 1 row (i=2)
    # Split columns of B: left 2 cols (k=0,1) and right 2 cols (k=2,3)
    # Split middle dim j: j=0,1 and j=2,3
    
    # For 2x2 * 2x2 * 2x2 block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]):
    # Use rank-7 Strassen-like decomposition
    # But we need to handle the full 3x4 * 4x4 structure
    
    # Let's construct a structured decomposition:
    # 
    # The tensor T[i,j,k] = 1 for the multiplication tensor (C[i,k] = sum_j A[i,j]*B[j,k])
    # We want to decompose this using the rank-7 structure for 2x2 blocks.
    #
    # For the 2x2 * 2x2 -> 2x2 case, the rank-7 decomposition has terms:
    # Each term r has:
    #   u_r[i,j] for i,j in {0,1} (A factor)
    #   v_r[j,k] for j,k in {0,1} (B factor)  
    #   w_r[i,k] for i,k in {0,1} (C factor)
    #
    # Standard rank-7 2x2 decomposition (one known form):
    # Term 0: u=[[1,0],[0,0]], v=[[1,1],[0,0]], w=[[1,0],[0,0]]  -> a11*(b11+b12) -> c11
    # Term 1: u=[[0,1],[0,0]], v=[[0,0],[1,1]], w=[[1,0],[0,0]]  -> a12*(b21+b22) -> c11
    # Term 2: u=[[0,0],[1,0]], v=[[1,0],[0,1]], w=[[0,0],[1,0]]  -> a21*(b11+b22) -> c21
    # Term 3: u=[[0,0],[0,1]], v=[[0,1],[1,0]], w=[[0,0],[1,0]]  -> a22*(b12+b21) -> c21
    # Term 4: u=[[1,0],[0,0]], v=[[0,0],[0,1]], w=[[1,0],[0,0]]  -> a11*b22 -> c11
    # Term 5: u=[[0,0],[0,1]], v=[[1,0],[0,0]], w=[[0,0],[1,0]]  -> a22*b11 -> c21
    # Term 6: u=[[0,1],[0,0]], v=[[0,0],[1,0]], w=[[1,0],[0,0]]  -> a12*b21 -> c11
    #         u=[[0,0],[1,0]], v=[[0,1],[0,0]], w=[[0,0],[1,0]]  -> a21*b12 -> c21
    #         These combine as a12*b21 - a21*b12 contributing to c11 and c21 with signs
    #
    # Actually, let me use a cleaner known rank-7 decomposition.
    # The 2x2 matrix multiplication tensor has rank 7. One explicit decomposition:
    #
    # Let's denote the 8 schoolbook terms as (i,j,k) for A[i,j]*B[j,k]->C[i,k]
    # Rank-7 combines these 8 terms into 7.
    #
    # A well-known rank-7 decomposition (Bini et al. / Strassen variant):
    # m1 = a11*(b11+b12)
    # m2 = a12*(b21+b22)
    # m3 = a21*(b11+b22)
    # m4 = a22*(b21+b12)
    # m5 = a11*b22
    # m6 = a22*b11
    # m7 = (a12-a21)*(b21-b12)  [or similar]
    #
    # C11 = m1 + m2 - m5 - m7
    # C12 = m3 + m4 - m6 - m7  [wait, need to be careful]
    # C21 = m3 + m4 - m5 - m7
    # C22 = m1 + m2 - m6 - m7
    #
    # Let me verify: 
    # C11 should be a11*b11 + a12*b21
    # m1 + m2 - m5 - m7 = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22 - (a12-a21)*(b21-b12)
    # = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22 - a12*b21 + a12*b12 + a21*b21 - a21*b12
    # = a11*b11 + a11*b12 + a12*b22 - a11*b22 + a12*b12 + a21*b21 - a21*b12
    # This doesn't simplify to a11*b11 + a12*b21. So this particular formula is wrong.
    #
    # Let me use a verified rank-7 decomposition. The standard one:
    #
    # Actually, for the purpose of this task, let me try a different approach.
    # I'll use the fact that 3x4 * 4x4 can be seen as a sum of rank-1 terms
    # and try to find a structured low-rank decomposition.
    #
    # Alternative approach: Use the Kronecker product structure directly.
    # The multiplication tensor for (n,m,p) can be decomposed using:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (where these are basis vectors)
    # This gives rank n*m*p = 48.
    #
    # To get below 48, we need to exploit linear dependencies.
    # For 2x2x2, the tensor rank is 7 (not 8).
    #
    # For 3x4x4, let's try to use the 2x2 rank-7 structure.
    # 
    # Split the problem into blocks:
    # A (3x4) = [A_top (2x4); A_bot (1x4)]
    # B (4x4) = [B_left (4x2) | B_right (4x2)]
    #
    # C = A*B (3x4)
    # C[0:2, 0:2] = A_top * B_left  (2x4 * 4x2 -> 2x2)
    # C[0:2, 2:4] = A_top * B_right  (2x4 * 4x2 -> 2x2)
    # C[2:3, 0:2] = A_bot * B_left  (1x4 * 4x2 -> 1x2)
    # C[2:3, 2:4] = A_bot * B_right  (1x4 * 4x2 -> 1x2)
    #
    # For 2x4 * 4x2 -> 2x2:
    # Split the middle dimension 4 into 2+2:
    # A_top = [A1 (2x2) | A2 (2x2)]
    # B_left = [B1 (2x2); B2 (2x2)]
    # C_top_left = A1*B1 + A2*B2
    # Each 2x2*2x2->2x2 has rank 7, so total rank 14 for this block.
    #
    # Similarly C_top_right = A1*B3 + A2*B4 where B3, B4 are the two 2x2 blocks of B_right
    # Rank 14.
    #
    # For 1x4 * 4x2 -> 1x2:
    # A_bot = [a1 (1x2) | a2 (1x2)]
    # B_left = [B1 (2x2); B2 (2x2)]
    # C_bot_left = a1*B1 + a2*B2
    # Each 1x2*2x2->1x2 has rank 2 (schoolbook), so total rank 4.
    #
    # Similarly C_bot_right has rank 4.
    #
    # Total rank: 14 + 14 + 4 + 4 = 36.
    #
    # Let me implement this structured decomposition with rank 36.

    # Rank-7 2x2*2x2 decomposition:
    # For A=[[a11,a12],[a21,a22]], B=[[b11,b12],[b21,b22]], C=A*B
    # The 7 bilinear forms and their contributions to C:
    #
    # I'll use the following verified rank-7 decomposition:
    # f1 = a11 * (b11 + b12)
    # f2 = a12 * (b21 + b22)
    # f3 = a21 * (b11 + b22)
    # f4 = a22 * (b21 + b12)
    # f5 = a11 * b22
    # f6 = a22 * b11
    # f7 = a12 * b21 - a21 * b12
    #
    # C[0,0] = f1 + f2 - f5 - f7
    # C[0,1] = f3 + f4 - f6 - f7
    # C[1,0] = f3 + f4 - f5 - f7
    # C[1,1] = f1 + f2 - f6 - f7
    #
    # Let me verify C[0,0]:
    # f1 + f2 - f5 - f7 = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22 - (a12*b21 - a21*b12)
    # = a11*b11 + a11*b12 + a12*b21 + a12*b22 - a11*b22 - a12*b21 + a21*b12
    # = a11*b11 + a11*b12 + a12*b22 - a11*b22 + a21*b12
    # This should equal a11*b11 + a12*b21. It doesn't. So this decomposition is incorrect.
    #
    # Let me look up the correct rank-7 2x2 decomposition.
    # 
    # Correct rank-7 decomposition (from literature):
    # The 2x2 matrix multiplication tensor has rank 7. One explicit decomposition:
    #
    # Using the notation where each term is (u, v, w) with u in A-space, v in B-space, w in C-space:
    #
    # Term 1: u = e11, v = e11 + e12, w = e11
    # Term 2: u = e12, v = e21 + e22, w = e11
    # Term 3: u = e21, v = e11 + e22, w = e21
    # Term 4: u = e22, v = e21 + e12, w = e21
    # Term 5: u = e11, v = e22, w = e11
    # Term 6: u = e22, v = e11, w = e21
    # Term 7: u = e12 - e21, v = e21 - e12, w = e11 + e21
    #
    # Wait, I need to be more careful. Let me just use a known correct formulation.
    #
    # Actually, for the purposes of this task, let me try a simpler approach:
    # Use the schoolbook decomposition but check if we can merge any terms.
    #
    # Given the complexity, let me just return the schoolbook decomposition for now,
    # but structure the code so it's clear where improvements could be made.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
