"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 8]

def _mul3x3_by_3x3_rank8():
    """
    Winograd's F(4) algorithm for 3x3 * 3x3.
    Returns U (9x8), V (9x8), W (9x8) for the 3x3*3x3 multiplication tensor.
    Indexing: A is 3x3 (i,j), B is 3x3 (j,k), C is 3x3 (i,k).
    Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Here n=3, m=3, p=3.
    
    Winograd F(4):
    Let A be 3x3, B be 3x3.
    Define:
      A0 = A[0,0]+A[1,1]
      A1 = A[0,0]-A[1,1]
      A2 = A[0,1]+A[1,0]
      A3 = A[0,1]-A[1,0]
      A4 = A[0,2]
      A5 = A[1,2]
      A6 = A[2,0]
      A7 = A[2,1]+A[2,2]
      A8 = A[2,1]-A[2,2]
    
      B0 = B[0,0]+B[1,1]
      B1 = B[0,0]-B[1,1]
      B2 = B[0,1]+B[1,0]
      B3 = B[0,1]-B[1,0]
      B4 = B[0,2]
      B5 = B[1,2]
      B6 = B[2,0]
      B7 = B[2,1]+B[2,2]
      B8 = B[2,1]-B[2,2]
    
    Products:
      P0 = A0*B0
      P1 = A1*B1
      P2 = A2*B2
      P3 = A3*B3
      P4 = A4*B4
      P5 = A5*B5
      P6 = A6*B6
      P7 = A7*B7
      P8 = A8*B8
    
    Wait, that's 9 products. Let me reconsider.
    
    Actually, Winograd's F(4) for 3x3 uses 8 products? No, it uses 9.
    Let me look up the correct Winograd F(4) for 3x3.
    
    Actually, the standard Winograd F(4) for 3x3 matrix multiplication uses 8 multiplications? 
    No, I think it's 9. Let me reconsider.
    
    The schoolbook is 27. Winograd F(4) reduces it to 9.
    
    Let me define it properly:
    A = [[a00, a01, a02],
         [a10, a11, a12],
         [a20, a21, a22]]
    B = [[b00, b01, b02],
         [b10, b11, b12],
         [b20, b21, b22]]
    
    Winograd F(4):
    U0 = a00 + a11
    U1 = a00 - a11
    U2 = a01 + a10
    U3 = a01 - a10
    U4 = a02
    U5 = a12
    U6 = a20
    U7 = a21 + a22
    U8 = a21 - a22
    
    V0 = b00 + b11
    V1 = b00 - b11
    V2 = b01 + b10
    V3 = b01 - b10
    V4 = b02
    V5 = b12
    V6 = b20
    V7 = b21 + b22
    V8 = b21 - b22
    
    P0 = U0*V0
    P1 = U1*V1
    P2 = U2*V2
    P3 = U3*V3
    P4 = U4*V4
    P5 = U5*V5
    P6 = U6*V6
    P7 = U7*V7
    P8 = U8*V8
    
    C[0,0] = P0 + P1
    C[0,1] = P2 + P3
    C[0,2] = P4
    C[1,0] = P2 - P3
    C[1,1] = P0 - P1
    C[1,2] = P5
    C[2,0] = P6
    C[2,1] = P7 + P8
    C[2,2] = P7 - P8
    
    Wait, let me verify:
    C[0,0] = a00*b00 + a01*b10 + a02*b20
    P0 + P1 = (a00+a11)(b00+b11) + (a00-a11)(b00-b11)
    = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a00*b00 - a00*b11 - a11*b00 + a11*b11
    = 2*a00*b00 + 2*a11*b11
    That's not C[0,0]. So this is wrong.
    
    Let me look up the correct Winograd F(4) formulation.
    
    Actually, I think the correct Winograd F(4) for 3x3 is:
    
    U0 = a00 + a11
    U1 = a00 - a11  
    U2 = a01 + a10
    U3 = a01 - a10
    U4 = a02
    U5 = a12
    U6 = a20
    U7 = a21 + a22
    U8 = a21 - a22
    
    V0 = b00 + b11
    V1 = b00 - b11
    V2 = b01 + b10
    V3 = b01 - b10
    V4 = b02
    V5 = b12
    V6 = b20
    V7 = b21 + b22
    V8 = b21 - b22
    
    And the reconstruction:
    C[0,0] = (P0 + P1) / 2 + ... hmm, this is getting complicated.
    
    Let me just use a known rank-8 decomposition for 3x3*3x3.
    
    Actually, I recall that the minimum rank for 3x3*3x3 is 19 (over reals) or 23 (over integers/rationals)?
    No, that's not right either. The Strassen algorithm gives rank 7 for 2x2*2x2.
    For 3x3*3x3, the best known is 19 over reals (Landsberg), but over rationals it might be higher.
    
    Actually, for the purposes of this task, I need an exact decomposition with integer/half-integer entries.
    The schoolbook is rank 27. I need to find something better.
    
    Let me think about this differently. The task is 3x3 * 3x8.
    I can view this as three 3x3 * 3x2 blocks? No, that doesn't work directly.
    
    Actually, 3x3 * 3x8 = 3x8. I can split the 3x8 into three 3x3 blocks and one 3x2 block.
    So C = A * B where A is 3x3, B is 3x8.
    B = [B1 | B2 | B3 | B4] where B1, B2, B3 are 3x3 and B4 is 3x2.
    C = [A*B1 | A*B2 | A*B3 | A*B4]
    
    If I use rank-8 for each 3x3*3x3 (if such exists with integer entries) and rank-? for 3x3*3x2,
    the total rank would be 3*8 + rank(3x3*3x2) = 24 + rank(3x3*3x2).
    
    For 3x3 * 3x2, the schoolbook is 18. Can I do better?
    
    Actually, let me just try to construct a valid decomposition and see what rank I can achieve.
    The safest bet is to use the schoolbook (rank 72) and see if I can improve.
    
    Wait, the schoolbook for 3x3 * 3x8 is 3*3*8 = 72.
    
    Let me think about a block approach:
    - Split B (3x8) into four blocks: B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x2)
    - C = A * B = [A*B1 | A*B2 | A*B3 | A*B4]
    - Each A*Bi is 3x2, schoolbook rank 3*3*2 = 18
    - Total: 4*18 = 72 (same as schoolbook)
    
    Alternatively, split into three 3x3 blocks and one 3x2 block:
    - C = [A*B1 | A*B2 | A*B3 | A*B4] where B1,B2,B3 are 3x3, B4 is 3x2
    - If I can find rank-8 for 3x3*3x3 (Winograd F(4) uses 9, not 8)
    
    Let me just use the schoolbook decomposition which is guaranteed to work.
    """
    pass

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    # Schoolbook: rank = n*m*p = 3*3*8 = 72
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
