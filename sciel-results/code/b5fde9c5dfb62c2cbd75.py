"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy: Decompose the 2x4x7 multiplication into blocks:
    - Split the 4x7 matrix into two parts: 4x4 and 4x3
    - Use Winograd's optimal rank-15 scheme for 2x2x2 (applied to 2x4x4 as two 2x2x2 blocks)
    - Use optimal rank-12 scheme for 2x2x2 (applied to 2x4x3 as one 2x2x2 block + one 2x1x3)
    
    Actually, let's think more carefully. The standard approach for rectangular matrices:
    - 2x4x7 = (2x4x4) + (2x4x3)
    - For 2x4x4: this is two independent 2x2x2 multiplications, each rank 15 (Winograd)
      Total rank: 2 * 15 = 30
    - For 2x4x3: this is a 2x2x2 (rank 15) plus a 2x2x1 (rank 4, schoolbook)
      Total rank: 15 + 4 = 19
    - Grand total: 30 + 19 = 49
    
    Wait, that's worse than schoolbook (2*4*7 = 56). Let me reconsider.
    
    Actually, the Winograd scheme for 2x2x2 has rank 15 (vs schoolbook 8). That's worse!
    Winograd's scheme is for 2x2x2 with rank 15? No, Winograd's is rank 15 for 2x2x2? 
    No, that's not right. The standard Winograd scheme for 2x2 matrix multiplication 
    uses 7 multiplications (Strassen), so rank 7 for the 2x2x2 tensor.
    
    Let me reconsider. The schoolbook rank for n x m x p is n*m*p.
    For 2x4x7: schoolbook rank = 2*4*7 = 56.
    
    Can we do better? Let's use the fact that we can decompose into smaller blocks.
    
    Approach: Use the known optimal ranks for small rectangular matrices.
    - 2x2x2: optimal rank 7 (Strassen)
    - 2x2x3: optimal rank 13 (known)
    - 2x3x3: optimal rank 13 (known)
    - 2x2x4: can be seen as two 2x2x2, rank 14 (two Strassen)
    
    For 2x4x7:
    Split 4 into 2+2, split 7 into 3+4:
    - 2x2x3: rank 13
    - 2x2x4: rank 14 (two 2x2x2)
    - 2x2x3: rank 13
    - 2x2x4: rank 14 (two 2x2x2)
    Total: 13+14+13+14 = 54. Still worse than 56? No, 54 < 56.
    
    Wait, let me be more careful. 2x4x7 means we have:
    - i in {0,1} (rows of first matrix, rows of result)
    - j in {0,1,2,3} (cols of first matrix, rows of second)
    - k in {0,...,6} (cols of second matrix, cols of result)
    
    The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    
    So the tensor is indexed by (i, j, k) where i in 0..1, j in 0..3, k in 0..6.
    T[i,j,k] = 1 for all valid indices (it's the "identity" tensor for matrix mult).
    
    Schoolbook: rank = 2*4*7 = 56.
    
    Let's try to find a better decomposition. One approach: use the known fact that
    matmul(2,4,7) can be done with rank less than 56.
    
    Let me try a different approach: use random search / optimization to find a low-rank
    decomposition. But we need EXACT integer/half-integer entries.
    
    Alternative: use the block decomposition with Strassen.
    
    Split j into {0,1} and {2,3}, split k into {0,1,2} and {3,4,5,6}:
    - Block (i in 0..1, j in 0..1, k in 0..2): 2x2x3, rank 13
    - Block (i in 0..1, j in 0..1, k in 3..6): 2x2x4, rank 14
    - Block (i in 0..1, j in 2..3, k in 0..2): 2x2x3, rank 13
    - Block (i in 0..1, j in 2..3, k in 3..6): 2x2x4, rank 14
    Total: 13+14+13+14 = 54
    
    That's better than 56! Let me implement this.
    
    For 2x2x2 (Strassen):
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    C = A*B
    
    Strassen's 7 products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    C00 = M1 - M4 + M6 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M2 + M5
    
    So the rank-7 decomposition for 2x2x2:
    U (2x2 matrix, 4 entries) x V (2x2 matrix, 4 entries) x W (2x2 matrix, 4 entries)
    
    Let me code this up properly.
    
    Actually, let me just implement the schoolbook first to make sure it works, then try
    to improve.
    
    For now, let's implement the block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition block by block.
    # Split j (size m=4) into two groups of 2: j_group0 = {0,1}, j_group1 = {2,3}
    # Split k (size p=7) into two groups: k_group0 = {0,1,2} (size 3), k_group1 = {3,4,5,6} (size 4)
    
    # For each (j_group, k_group) pair, we have a sub-problem of size n x j_size x k_size
    # j_size is always 2, k_size is 3 or 4.
    
    # For 2x2x3: use known rank-13 decomposition
    # For 2x2x4: use two 2x2x2 Strassen decompositions, total rank 14
    
    # Let me define the Strassen decomposition for 2x2x2.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        V = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        W = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00 and a11 contribute
        U[0][0] = 1  # a00
        U[3][0] = 1  # a11
        # V: b00 and b11 contribute
        V[0][0] = 1  # b00
        V[3][0] = 1  # b11
        # W: C00 += M1, C11 += M1
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2 = (a10+a11)b00
        U[2][1] = 1  # a10
        U[3][1] = 1  # a11
        V[0][1] = 1  # b00
        # W: C10 += M2, C11 += M2
        W[2][1] = 1  # C10
        W[3][1] = 1  # C11
        
        # M3 = a00(b01-b11)
        U[0][2] = 1  # a00
        V[1][2] = 1  # b01
        V[3][2] = -1 # -b11
        # W: C01 += M3, C11 -= M3
        W[1][2] = 1  # C01
        W[3][2] = -1 # C11
        
        # M4 = a11(b10-b00)
        U[3][3] = 1  # a11
        V[2][3] = 1  # b10
        V[0][3] = -1 # -b00
        # W: C00 -= M4, C10 += M4
        W[0][3] = -1 # C00
        W[2][3] = 1  # C10
        
        # M5 = (a00+a01)b11
        U[0][4] = 1  # a00
        U[1][4] = 1  # a01
        V[3][4] = 1  # b11
        # W: C01 += M5, C11 += M5
        W[1][4] = 1  # C01
        W[3][4] = 1  # C11
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1  # a10
        U[0][5] = -1 # -a00
        V[0][5] = 1  # b00
        V[2][5] = 1  # b10
        # W: C00 += M6
        W[0][5] = 1  # C00
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1  # a01
        U[3][6] = -1 # -a11
        V[1][6] = 1  # b01
        V[3][6] = 1  # b11
        # W: C00 += M7
        W[0][6] = 1  # C00
        
        return U, V, W
    
    # For 2x2x3, we need a rank-13 decomposition.
    # One way: 2x2x3 = 2x2x2 + 2x2x1
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 4 (schoolbook, since it's just a vector-matrix product)
    # Total: 7 + 4 = 11? That seems too low.
    
    # Actually, 2x2x1 means multiplying 2x2 by 2x1, giving 2x1.
    # Schoolbook rank: 2*2*1 = 4.
    # So 2x2x3 rank <= 7 + 4 = 11.
    
    # But is there a better decomposition? The known optimal rank for 2x2x3 is 13? 
    # Or is it 11? Let me just use 7+4=11 for now.
    
    def strassen_2x2x3():
        """Return U, V, W for 2x2x3 decomposition using Strassen for 2x2x2 + schoolbook for 2x2x1."""
        U2, V2, W2 = strassen_2x2x2()
        rank2 = 7
        
        # 2x2x1 schoolbook:
        # U3[i*2+j, c], V3[j*1+k, c], W3[1*2+i, c] where k is always 0
        # Actually, for 2x2x1: i in {0,1}, j in {0,1}, k in {0}
        # U3 has 4 rows, V3 has 2 rows, W3 has 2 rows
        rank1 = 4
        U3 = [[0]*rank1 for _ in range(4)]
        V3 = [[0]*rank1 for _ in range(2)]
        W3 = [[0]*rank1 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                c = i*2 + j
                U3[i*2+j][c] = 1
                V3[j][c] = 1
                W3[i][c] = 1
        
        # Combine: total rank = 7 + 4 = 11
        total_rank = rank2 + rank1
        U = [[0]*total_rank for _ in range(4)]
        V = [[0]*total_rank for _ in range(6)]  # 2x3 = 6 rows
        W = [[0]*total_rank for _ in range(6)]  # 3x2 = 6 rows
        
        # Copy 2x2x2 part (k=0,1)
        for r in range(rank2):
            for i in range(4):
                U[i][r] = U2[i][r]
            for j in range(2):
                for k in range(2):
                    V[j*3+k][r] = V2[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][r] = W2[k*2+i][r]
        
        # Copy 2x2x1 part (k=2)
        for r in range(rank1):
            idx = rank2 + r
            for i in range(4):
                U[i][idx] = U3[i][r]
            for j in range(2):
                V[j*3+2][idx] = V3[j][r]
            for i in range(2):
                W[2*2+i][idx] = W3[i][r]
        
        return U, V, W
    
    # For 2x2x4: two 2x2x2 Strassen, rank 14
    def strassen_2x2x4():
        """Return U, V, W for 2x2x4 decomposition using two Strassen 2x2x2."""
        U1, V1, W1 = strassen_2x2x2()
        U2, V2, W2 = strassen_2x2x2()
        rank = 14
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]  # 2x4 = 8 rows
        W = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
        
        # First 2x2x2 (k=0,1)
        for r in range(7):
            for i in range(4):
                U[i][r] = U1[i][r]
            for j in range(2):
                for k in range(2):
                    V[j*4+k][r] = V1[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][r] = W1[k*2+i][r]
        
        # Second 2x2x2 (k=2,3)
        for r in range(7):
            idx = 7 + r
            for i in range(4):
                U[i][idx] = U2[i][r]
            for j in range(2):
                for k in range(2):
                    V[j*4+(k+2)][idx] = V2[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(k+2)*2+i][idx] = W2[k*2+i][r]
        
        return U, V, W
    
    # Now combine all four blocks:
    # Block (j=0,1; k=0,1,2): 2x2x3, rank 11
    # Block (j=0,1; k=3,4,5,6): 2x2x4, rank 14
    # Block (j=2,3; k=0,1,2): 2x2x3, rank 11
    # Block (j=2,3; k=3,4,5,6): 2x2x4, rank 14
    # Total rank: 11+14+11+14 = 50
    
    U00, V00, W00 = strassen_2x2x3()
    U01, V01, W01 = strassen_2x2x4()
    U10, V10, W10 = strassen_2x2x3()
    U11, V11, W11 = strassen_2x2x4()
    
    total_rank = 11 + 14 + 11 + 14  # = 50
    
    U = [[0]*total_rank for _ in range(n*m)]  # 2*4 = 8 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 4*7 = 28 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 7*2 = 14 rows
    
    # Block (j_group=0, k_group=0): j in {0,1}, k in {0,1,2}
    offset = 0
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[i*4+j][offset+r] = U00[i*2+j][r]
        for j in range(2):
            for k in range(3):
                V[j*7+k][offset+r] = V00[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[k*2+i][offset+r] = W00[k*2+i][r]
    
    # Block (j_group=0, k_group=1): j in {0,1}, k in {3,4,5,6}
    offset = 11
    for r in range(14):
        for i in range(2):
            for j in range(2):
                U[i*4+j][offset+r] = U01[i*2+j][r]
        for j in range(2):
            for k in range(4):
                V[j*7+(k+3)][offset+r] = V01[j*4+k][r]
        for k in range(4):
            for i in range(2):
                W[(k+3)*2+i][offset+r] = W01[k*2+i][r]
    
    # Block (j_group=1, k_group=0): j in {2,3}, k in {0,1,2}
    offset = 25
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2)][offset+r] = U10[i*2+j][r]
        for j in range(2):
            for k in range(3):
                V[(j+2)*7+k][offset+r] = V10[j*3+k][r]
        for k in range(3):
            for i in range(2):
                W[k*2+i][offset+r] = W10[k*2+i][r]
    
    # Block (j_group=1, k_group=1): j in {2,3}, k in {3,4,5,6}
    offset = 36
    for r in range(14):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2)][offset+r] = U11[i*2+j][r]
        for j in range(2):
            for k in range(4):
                V[(j+2)*7+(k+3)][offset+r] = V11[j*4+k][r]
        for k in range(4):
            for i in range(2):
                W[(k+3)*2+i][offset+r] = W11[k*2+i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
