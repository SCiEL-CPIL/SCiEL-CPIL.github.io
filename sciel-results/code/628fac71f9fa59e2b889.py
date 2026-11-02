import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We split the p=7 columns into p1=3 and p2=4.
    # Part 1: 3x3x3 using rank 24 (Strassen-based block decomposition).
    # Part 2: 3x3x4 using schoolbook rank 36.
    # Total rank 60.
    
    rank1 = 24
    rank2 = 3 * 3 * 4
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 3x3x3 (k=0,1,2)
    # We decompose C11, C12, C21, C22 where C is 3x3.
    # A = [[A22, a1], [a2T, a33]]
    # B = [[B22, b1], [b2T, b33]]
    # A22 is 2x2 (indices 0,1,3,4 in flattened 3x3: 0,1; 3,4)
    # Wait, flattened 3x3 index for A[i,j] is i*3+j.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # A22 corresponds to A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 3, 4
    # a1 corresponds to A[0,2], A[1,2] -> indices 2, 5
    # a2T corresponds to A[2,0], A[2,1] -> indices 6, 7
    # a33 corresponds to A[2,2] -> index 8
    
    # B is 3x3 sub-block (cols 0,1,2 of global B).
    # B[j,k] global index is j*p + k. Here p=7.
    # So B[0,0]=0, B[0,1]=1, B[0,2]=2
    # B[1,0]=7, B[1,1]=8, B[1,2]=9
    # B[2,0]=14, B[2,1]=15, B[2,2]=16
    
    # B22 corresponds to B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0, 1, 7, 8
    # b1 corresponds to B[0,2], B[1,2] -> indices 2, 9
    # b2T corresponds to B[2,0], B[2,1] -> indices 14, 15
    # b33 corresponds to B[2,2] -> index 16
    
    # C is 3x3 sub-block (rows 0,1,2 of global C, but C is p*n = 7*3 = 21).
    # W index is k*n + i.
    # C[k,i] global index is k*3 + i.
    # C11 (top-left 2x2):
    # C[0,0]=0, C[0,1]=1
    # C[1,0]=3, C[1,1]=4
    # C12 (top-right 2x1):
    # C[0,2]=2, C[1,2]=5
    # C21 (bottom-left 1x2):
    # C[2,0]=6, C[2,1]=7
    # C22 (bottom-right 1x1):
    # C[2,2]=8
    
    # Strassen 2x2x2 for A22 * B22
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b21-b22)
    # P4 = a22(b12-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    
    # Local indices for A22:
    # a11 -> A[0,0] -> 0
    # a12 -> A[0,1] -> 1
    # a21 -> A[1,0] -> 3
    # a22 -> A[1,1] -> 4
    
    # Local indices for B22:
    # b11 -> B[0,0] -> 0
    # b12 -> B[0,1] -> 1
    # b21 -> B[1,0] -> 7
    # b22 -> B[1,1] -> 8
    
    col_offset = 0
    
    # P1
    U[0][col_offset] += 1; U[4][col_offset] += 1
    V[0][col_offset] += 1; V[8][col_offset] += 1
    # C11 += P1, C22 += P1
    W[0][col_offset] += 1; W[8][col_offset] += 1
    col_offset += 1
    
    # P2
    U[3][col_offset] += 1; U[4][col_offset] += 1
    V[0][col_offset] += 1
    # C21 += P2
    W[3][col_offset] += 1
    # C22 -= P2
    W[8][col_offset] -= 1
    col_offset += 1
    
    # P3
    U[0][col_offset] += 1
    V[7][col_offset] += 1; V[8][col_offset] -= 1
    # C12 += P3
    W[1][col_offset] += 1
    # C22 += P3
    W[8][col_offset] += 1
    col_offset += 1
    
    # P4
    U[4][col_offset] += 1
    V[1][col_offset] += 1; V[0][col_offset] -= 1
    # C11 += P4
    W[0][col_offset] += 1
    # C21 += P4
    W[3][col_offset] += 1
    col_offset += 1
    
    # P5
    U[0][col_offset] += 1; U[1][col_offset] += 1
    V[8][col_offset] += 1
    # C11 -= P5
    W[0][col_offset] -= 1
    # C12 += P5
    W[1][col_offset] += 1
    col_offset += 1
    
    # P6
    U[3][col_offset] += 1; U[0][col_offset] -= 1
    V[0][col_offset] += 1; V[1][col_offset] += 1
    # C22 += P6
    W[8][col_offset] += 1
    col_offset += 1
    
    # P7
    U[1][col_offset] += 1; U[4][col_offset] -= 1
    V[7][col_offset] += 1; V[8][col_offset] += 1
    # C11 += P7
    W[0][col_offset] += 1
    col_offset += 1
    
    # Now the cross terms.
    # C11 += a1 * b2T
    # a1: A[0,2] (idx 2), A[1,2] (idx 5)
    # b2T: B[2,0] (idx 14), B[2,1] (idx 15)
    # This is a 2x1 * 1x2 matrix product. Rank 2.
    # Term 1: A[0,2] * B[2,0] -> C[0,0]
    U[2][col_offset] = 1; V[14][col_offset] = 1; W[0][col_offset] = 1
    col_offset += 1
    # Term 2: A[0,2] * B[2,1] -> C[0,1]
    U[2][col_offset] = 1; V[15][col_offset] = 1; W[1][col_offset] = 1
    col_offset += 1
    # Term 3: A[1,2] * B[2,0] -> C[1,0]
    U[5][col_offset] = 1; V[14][col_offset] = 1; W[3][col_offset] = 1
    col_offset += 1
    # Term 4: A[1,2] * B[2,1] -> C[1,1]
    U[5][col_offset] = 1; V[15][col_offset] = 1; W[4][col_offset] = 1
    col_offset += 1
    
    # C12 += A22 * b1
    # A22: 2x2. b1: B[0,2] (idx 2), B[1,2] (idx 9).
    # This is 2x2 * 2x1. Rank 4.
    # C[0,2] += A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] += A[1,0]B[0,2] + A[1,1]B[1,2]
    U[0][col_offset] = 1; V[2][col_offset] = 1; W[2][col_offset] = 1
    col_offset += 1
    U[1][col_offset] = 1; V[9][col_offset] = 1; W[2][col_offset] = 1
    col_offset += 1
    U[3][col_offset] = 1; V[2][col_offset] = 1; W[5][col_offset] = 1
    col_offset += 1
    U[4][col_offset] = 1; V[9][col_offset] = 1; W[5][col_offset] = 1
    col_offset += 1
    
    # C12 += a1 * b33
    # a1: A[0,2], A[1,2]. b33: B[2,2] (idx 16).
    # Rank 2.
    # C[0,2] += A[0,2]B[2,2]
    U[2][col_offset] = 1; V[16][col_offset] = 1; W[2][col_offset] = 1
    col_offset += 1
    # C[1,2] += A[1,2]B[2,2]
    U[5][col_offset] = 1; V[16][col_offset] = 1; W[5][col_offset] = 1
    col_offset += 1
    
    # C21 += a2T * B22
    # a2T: A[2,0] (idx 6), A[2,1] (idx 7).
    # B22: 2x2.
    # Rank 4.
    # C[2,0] += A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] += A[2,0]B[0,1] + A[2,1]B[1,1]
    U[6][col_offset] = 1; V[0][col_offset] = 1; W[6][col_offset] = 1
    col_offset += 1
    U[7][col_offset] = 1; V[7][col_offset] = 1; W[6][col_offset] = 1
    col_offset += 1
    U[6][col_offset] = 1; V[1][col_offset] = 1; W[7][col_offset] = 1
    col_offset += 1
    U[7][col_offset] = 1; V[8][col_offset] = 1; W[7][col_offset] = 1
    col_offset += 1
    
    # C21 += a33 * b2T
    # a33: A[2,2] (idx 8).
    # b2T: B[2,0] (idx 14), B[2,1] (idx 15).
    # Rank 2.
    # C[2,0] += A[2,2]B[2,0]
    U[8][col_offset] = 1; V[14][col_offset] = 1; W[6][col_offset] = 1
    col_offset += 1
    # C[2,1] += A[2,2]B[2,1]
    U[8][col_offset] = 1; V[15][col_offset] = 1; W[7][col_offset] = 1
    col_offset += 1
    
    # C22 += a2T * b1
    # a2T: A[2,0], A[2,1].
    # b1: B[0,2] (idx 2), B[1,2] (idx 9).
    # Rank 2.
    # C[2,2] += A[2,0]B[0,2] + A[2,1]B[1,2]
    U[6][col_offset] = 1; V[2][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    U[7][col_offset] = 1; V[9][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    
    # C22 += a33 * b33
    # Rank 1.
    U[8][col_offset] = 1; V[16][col_offset] = 1; W[8][col_offset] = 1
    col_offset += 1
    
    # Verify col_offset is 24
    # 7 (Strassen) + 4 (a1b2T) + 4 (A22b1) + 2 (a1b33) + 4 (a2TB22) + 2 (a33b2T) + 2 (a2Tb1) + 1 (a33b33) = 26?
    # Let's recount.
    # Strassen: 7
    # a1*b2T: 4 terms (2x2 outer product)
    # A22*b1: 4 terms (2x2 * 2x1)
    # a1*b33: 2 terms (2x1 * 1x1)
    # a2T*B22: 4 terms (1x2 * 2x2)
    # a33*b2T: 2 terms (1x1 * 1x2)
    # a2T*b1: 2 terms (1x2 * 2x1)
    # a33*b33: 1 term
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    
    # My col_offset calculation:
    # P1..P7: 7
    # a1b2T: 4
    # A22b1: 4
    # a1b33: 2
    # a2TB22: 4
    # a33b2T: 2
    # a2Tb1: 2
    # a33b33: 1
    # Total 26.
    
    # I allocated rank1 = 24. This is a bug in my plan.
    # I must set rank1 = 26.
    
    # Let's re-verify the rank of a1*b2T.
    # a1 is 2x1, b2T is 1x2. Product is 2x2.
    # $C_{11} += a_{13} b_{23}^T$.
    # $C_{11}$ is 2x2.
    # $a_{13} = [A_{0,2}, A_{1,2}]^T$.
    # $b_{23}^T = [B_{2,0}, B_{2,1}]$.
    # Product:
    # $[A_{0,2}B_{2,0}, A_{0,2}B_{2,1}]$
    # $[A_{1,2}B_{2,0}, A_{1,2}B_{2,1}]$
    # Yes, 4 terms.
    
    # So rank1 is 26.
    # Total rank = 26 + 36 = 62.
    # This is still better than 63.

    # Part 2: 3x3x4 (k=3,4,5,6)
    # Schoolbook for B[:, 3:7]
    # B indices: j*p + k for k in 3..6.
    # C indices: k*n + i for k in 3..6.
    
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    # Re-define construct to use the logic above, but with correct rank allocation.
    n, m, p = DIMENSIONS
    rank1 = 26
    rank2 = 3 * 3 * 4
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Part 1: 3x3x3 block (cols 0,1,2 of B)
    # Strassen 2x2x2
    # P1
    U[0][col_offset] += 1; U[4][col_offset] += 1
    V[0][col_offset] += 1; V[8][col_offset] += 1
    W[0][col_offset] += 1; W[8][col_offset] += 1
    col_offset += 1
    # P2
    U[3][col_offset] += 1; U[4][col_offset] += 1
    V[0][col_offset] += 1
    W[3][col_offset] += 1
    W[8][col_offset] -= 1
    col_offset += 1
    # P3
    U[0][col_offset] += 1
    V[7][col_offset] += 1; V[8][col_offset] -= 1
    W[1][col_offset] += 1
    W[8][col_offset] += 1
    col_offset += 1
    # P4
    U[4][col_offset] += 1
    V[1][col_offset] += 1; V[0][col_offset] -= 1
    W[0][col_offset] += 1
    W[3][col_offset] += 1
    col_offset += 1
    # P5
    U[0][col_offset] += 1; U[1][col_offset] += 1
    V[8][col_offset] += 1
    W[0][col_offset] -= 1
    W[1][col_offset] += 1
    col_offset += 1
    # P6
    U[3][col_offset] += 1; U[0][col_offset] -= 1
    V[0][col_offset] += 1; V[1][col_offset] += 1
    W[8][col_offset] += 1
    col_offset += 1
    # P7
    U[1][col_offset] += 1; U[4][col_offset] -= 1
    V[7][col_offset] += 1; V[8][col_offset] += 1
    W[0][col_offset] += 1
    col_offset += 1
    
    # a1 * b2T (4 terms)
    # a1: A[0,2] (2), A[1,2] (5)
    # b2T: B[2,0] (14), B[2,1] (15)
    U[2][col_offset] = 1; V[14][col_offset] = 1; W[0][col_offset] = 1; col_offset += 1
    U[2][col_offset] = 1; V[15][col_offset] = 1; W[1][col_offset] = 1; col_offset += 1
    U[5][col_offset] = 1; V[14][col_offset] = 1; W[3][col_offset] = 1; col_offset += 1
    U[5][col_offset] = 1; V[15][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    
    # A22 * b1 (4 terms)
    # A22: 0,1,3,4. b1: B[0,2] (2), B[1,2] (9)
    U[0][col_offset] = 1; V[2][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    U[1][col_offset] = 1; V[9][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    U[3][col_offset] = 1; V[2][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    U[4][col_offset] = 1; V[9][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    
    # a1 * b33 (2 terms)
    # a1: 2,5. b33: B[2,2] (16)
    U[2][col_offset] = 1; V[16][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    U[5][col_offset] = 1; V[16][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    
    # a2T * B22 (4 terms)
    # a2T: A[2,0] (6), A[2,1] (7). B22: 0,1,7,8
    U[6][col_offset] = 1; V[0][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    U[7][col_offset] = 1; V[7][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    U[6][col_offset] = 1; V[1][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    U[7][col_offset] = 1; V[8][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    
    # a33 * b2T (2 terms)
    # a33: A[2,2] (8). b2T: 14,15
    U[8][col_offset] = 1; V[14][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    U[8][col_offset] = 1; V[15][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    
    # a2T * b1 (2 terms)
    # a2T: 6,7. b1: 2,9
    U[6][col_offset] = 1; V[2][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
    U[7][col_offset] = 1; V[9][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
    
    # a33 * b33 (1 term)
    U[8][col_offset] = 1; V[16][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
    
    # Part 2: 3x3x4 block (cols 3,4,5,6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
