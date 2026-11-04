"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p = 112 (schoolbook).
    # We want to find a lower rank decomposition.
    
    # Key insight: For 4x4 * 4x4, we can use Strassen-like decompositions.
    # The best known rank for 4x4*4x4 is 49 (from Strassen's algorithm applied to 2x2 blocks).
    # For 4x4 * 4x7, we can decompose B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    # We can use rank-49 decomposition for A*B1 and rank-48 schoolbook for A*B2.
    # Total rank: 49 + 48 = 97.
    
    # But we can do better! Let's think about this more carefully.
    # Actually, the tensor structure is:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k)
    
    # For 4x4*4x4, Strassen gives rank 49. Let's construct that.
    # Then for the remaining 3 columns of B, we add 4*4*3 = 48 schoolbook terms.
    # Total: 49 + 48 = 97.
    
    # Let's construct the Strassen decomposition for 4x4*4x4.
    # Strassen's algorithm for 2x2*2x2 uses 7 multiplications.
    # For 4x4*4x4, we split into 2x2 blocks and apply Strassen to each.
    # This gives rank 7*7 = 49.
    
    # Strassen's 2x2*2x2 decomposition:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For 4x4*4x4, we have 4x4 = (2x2)^2, so we apply Strassen in both dimensions.
    # This gives 7*7 = 49 terms.
    
    # Let's construct the 4x4*4x4 Strassen decomposition explicitly.
    # A is 4x4, split into 4 blocks of 2x2:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = A*B, C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each 2x2*2x2 product, we use Strassen with 7 terms.
    # So C11 uses 7 terms (for A11*B11) + 7 terms (for A12*B21) = 14 terms
    # Similarly for C12, C21, C22.
    # Total: 4 * 14 = 56? No, that's not right.
    
    # Actually, Strassen's algorithm for 4x4*4x4 works differently.
    # We can view 4x4*4x4 as (2x2)^2 * (2x2)^2.
    # The Kronecker product of two Strassen tensors gives rank 7*7 = 49.
    
    # Let me construct this more carefully using the Kronecker product approach.
    # The Strassen tensor for 2x2*2x2 has rank 7.
    # The 4x4*4x4 tensor is the Kronecker product of two 2x2*2x2 tensors.
    # Its rank is at most 7*7 = 49.
    
    # Let's define the Strassen factors for 2x2*2x2:
    # U2 has shape (2*2, 7) = (4, 7)
    # V2 has shape (2*2, 7) = (4, 7)  
    # W2 has shape (2*2, 7) = (4, 7)
    
    # For 2x2*2x2:
    # U2[i*2+j, c] for i,j in {0,1}, c in {0,...,6}
    # V2[j*2+k, c] for j,k in {0,1}, c in {0,...,6}
    # W2[k*2+i, c] for k,i in {0,1}, c in {0,...,6}
    
    # Strassen's 7 terms:
    # m1: (a11+a22)(b11+b22)
    # m2: (a21+a22)b11
    # m3: a11(b12-b22)
    # m4: a22(b21-b11)
    # m5: (a11+a12)b22
    # m6: (a21-a11)(b11+b12)
    # m7: (a12-a22)(b21+b22)
    
    # Let me index A as A[i,j] with i,j in {0,1}
    # A[0,0]=a11, A[0,1]=a12, A[1,0]=a21, A[1,1]=a22
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1: (a11+a22)(b11+b22)
    # U2: A[0,0]+A[1,1] => U2[0,0]=1, U2[3,0]=1
    # V2: B[0,0]+B[1,1] => V2[0,0]=1, V2[3,0]=1
    # W2: C[0,0]+C[1,1] => W2[0,0]=1, W2[3,0]=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # m2: (a21+a22)b11
    # U2: A[1,0]+A[1,1] => U2[2,1]=1, U2[3,1]=1
    # V2: B[0,0] => V2[0,1]=1
    # W2: C[1,0] => W2[2,1]=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # m3: a11(b12-b22)
    # U2: A[0,0] => U2[0,2]=1
    # V2: B[0,1]-B[1,1] => V2[1,2]=1, V2[3,2]=-1
    # W2: C[0,1] => W2[1,2]=1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # m4: a22(b21-b11)
    # U2: A[1,1] => U2[3,3]=1
    # V2: B[1,0]-B[0,0] => V2[2,3]=1, V2[0,3]=-1
    # W2: C[1,0] => W2[2,3]=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1
    
    # m5: (a11+a12)b22
    # U2: A[0,0]+A[0,1] => U2[0,4]=1, U2[1,4]=1
    # V2: B[1,1] => V2[3,4]=1
    # W2: C[0,0] => W2[0,4]=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1
    
    # m6: (a21-a11)(b11+b12)
    # U2: A[1,0]-A[0,0] => U2[2,5]=1, U2[0,5]=-1
    # V2: B[0,0]+B[0,1] => V2[0,5]=1, V2[1,5]=1
    # W2: C[1,1] => W2[3,5]=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # m7: (a12-a22)(b21+b22)
    # U2: A[0,1]-A[1,1] => U2[1,6]=1, U2[3,6]=-1
    # V2: B[1,0]+B[1,1] => V2[2,6]=1, V2[3,6]=1
    # W2: C[0,1] => W2[1,6]=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[1][6] = 1
    
    # Now construct 4x4*4x4 using Kronecker product.
    # A4[i,j] where i,j in {0,1,2,3}
    # i = 2*i1 + i2, j = 2*j1 + j2 where i1,j1 in {0,1} (block indices), i2,j2 in {0,1} (within block)
    # Similarly for B and C.
    
    # U4 has shape (4*4, 49) = (16, 49)
    # V4 has shape (4*4, 49) = (16, 49)
    # W4 has shape (4*4, 49) = (16, 49)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    flat_A = (2*i1 + i2)*4 + (2*j1 + j2)
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            # U4[flat_A, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            U4[flat_A][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    flat_B = (2*j1 + j2)*4 + (2*k1 + k2)
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            V4[flat_B][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    flat_C = (2*k1 + k2)*4 + (2*i1 + i2)
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            W4[flat_C][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    # Now extend to 4x4*4x7.
    # B is 4x7, so we have 4*7 = 28 entries.
    # The first 16 entries correspond to the 4x4 part (columns 0-3)
    # The last 12 entries correspond to columns 4-6
    
    # For the extension, use schoolbook for the extra 3 columns.
    # This adds 4*4*3 = 48 more terms.
    
    total_rank = 49 + 4*4*3  # 49 + 48 = 97
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Copy the 4x4x4 part
    for row in range(16):
        for col in range(49):
            U[row][col] = U4[row][col]
            V[row][col] = V4[row][col]
            W[row][col] = W4[row][col]
    
    # Add schoolbook terms for columns 4-6 of B
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_actual = k + 4  # columns 4, 5, 6
                U[i*4 + j][col_offset] = 1
                V[j*7 + k_actual][col_offset] = 1
                W[k_actual*4 + i][col_offset] = 1
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
