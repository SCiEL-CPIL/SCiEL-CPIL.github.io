"""
Matrix multiplication 2x4 @ 4x6: find a rank-minimal exact tensor decomposition.

Strategy:
1. Build a schoolbook decomposition (rank = 48) as a safe fallback.
2. Construct a Strassen-based decomposition with rank 42 using block partitioning.
3. Verify correctness and return the best decomposition found.

The output uses the official trace convention W[k*n+i, r].
U has shape (n*m, rank) = (8, rank)
V has shape (m*p, rank) = (24, rank)
W has shape (p*n, rank) = (12, rank)
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before kill

    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (12, rank)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def verify(U, V, W):
        rank = len(U[0]) if U and U[0] else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += float(U[i * m + j][r]) * float(V[j * p + k][r]) * float(W[k * n + i][r])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # First, get the schoolbook solution as fallback
    best_result, best_rank = schoolbook()

    # Build a Strassen-based decomposition with rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(offset, A_rows, B_rows, C_rows):
        """
        Add a Strassen 2x2x2 product at the given offset.
        
        A_rows: [a00, a01, a10, a11] - U row indices for the 2x2 A block
        B_rows: [b00, b01, b10, b11] - V row indices for the 2x2 B block
        C_rows: [c00, c01, c10, c11] - W row indices for the 2x2 C block
        """
        a00, a01, a10, a11 = A_rows
        b00, b01, b10, b11 = B_rows
        c00, c01, c10, c11 = C_rows

        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        V[b11][offset] += 1
        W[c00][offset] += 1
        W[c11][offset] += 1

        # M2 = (a10 + a11) * b00
        U[a10][offset + 1] += 1
        U[a11][offset + 1] += 1
        V[b00][offset + 1] += 1
        W[c10][offset + 1] += 1

        # M3 = a00 * (b01 - b11)
        U[a00][offset + 2] += 1
        V[b01][offset + 2] += 1
        V[b11][offset + 2] += -1
        W[c01][offset + 2] += 1
        W[c11][offset + 2] += 1

        # M4 = a11 * (b10 - b00)
        U[a11][offset + 3] += 1
        V[b10][offset + 3] += 1
        V[b00][offset + 3] += -1
        W[c00][offset + 3] += 1
        W[c10][offset + 3] += 1

        # M5 = (a00 + a01) * b11
        U[a00][offset + 4] += 1
        U[a01][offset + 4] += 1
        V[b11][offset + 4] += 1
        W[c00][offset + 4] += -1
        W[c01][offset + 4] += 1

        # M6 = (a10 - a00) * (b00 + b01)
        U[a10][offset + 5] += 1
        U[a00][offset + 5] += -1
        V[b00][offset + 5] += 1
        V[b01][offset + 5] += 1
        W[c11][offset + 5] += 1

        # M7 = (a01 - a11) * (b10 + b11)
        U[a01][offset + 6] += 1
        U[a11][offset + 6] += -1
        V[b10][offset + 6] += 1
        V[b11][offset + 6] += 1
        W[c00][offset + 6] += 1
        W[c11][offset + 6] += -1

        return offset + 7

    # Define block indices
    # A is 2x4, split into two 2x2 blocks horizontally:
    # A_L: rows 0,1 cols 0,1 -> U indices [0, 1, 2, 3] in row-major
    # Wait, let me be more careful.
    # A[i,j] corresponds to U[i*m + j]
    # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[4], A[0,3] -> U[5]
    # A[1,0] -> U[2], A[1,1] -> U[3], A[1,2] -> U[6], A[1,3] -> U[7]
    
    # Let me re-examine the indexing:
    # U has shape (n*m, rank) = (8, rank)
    # U[i*m + j] corresponds to A[i,j]
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[4], A[0,3]=U[5]  -- Wait, this is wrong.
    # Actually i*m + j: i=0,j=0->0; i=0,j=1->1; i=0,j=2->2; i=0,j=3->3
    # i=1,j=0->4; i=1,j=1->5; i=1,j=2->6; i=1,j=3->7
    
    # So:
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    # A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    
    # A_L (left 2x2 block of A): [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A_L = [[U[0], U[1]], [U[4], U[5]]]
    # In Strassen order: a00=A[0,0]=U[0], a01=A[0,1]=U[1], a10=A[1,0]=U[4], a11=A[1,1]=U[5]
    A_L = [0, 1, 4, 5]
    
    # A_R (right 2x2 block of A): [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A_R = [[U[2], U[3]], [U[6], U[7]]]
    # In Strassen order: a00=A[0,2]=U[2], a01=A[0,3]=U[3], a10=A[1,2]=U[6], a11=A[1,3]=U[7]
    A_R = [2, 3, 6, 7]

    # B is 4x6, split into a 2x3 grid of 2x2 blocks vertically and horizontally:
    # B has 4 rows and 6 columns.
    # Top half: rows 0,1; Bottom half: rows 2,3
    # Columns split into three groups of 2: cols 0-1, 2-3, 4-5
    
    # V[j*p + k] corresponds to B[j,k]
    # j=0,k=0->0; j=0,k=1->1; j=0,k=2->2; j=0,k=3->3; j=0,k=4->4; j=0,k=5->5
    # j=1,k=0->6; j=1,k=1->7; j=1,k=2->8; j=1,k=3->9; j=1,k=4->10; j=1,k=5->11
    # j=2,k=0->12; j=2,k=1->13; j=2,k=2->14; j=2,k=3->15; j=2,k=4->16; j=2,k=5->17
    # j=3,k=0->18; j=3,k=1->19; j=3,k=2->20; j=3,k=3->21; j=3,k=4->22; j=3,k=5->23
    
    # B_T1 (top, left): rows 0,1 cols 0,1
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[6], B[1,1]=V[7]
    B_T1 = [0, 1, 6, 7]
    
    # B_T2 (top, middle): rows 0,1 cols 2,3
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[8], B[1,3]=V[9]
    B_T2 = [2, 3, 8, 9]
    
    # B_T3 (top, right): rows 0,1 cols 4,5
    # B[0,4]=V[4], B[0,5]=V[5], B[1,4]=V[10], B[1,5]=V[11]
    B_T3 = [4, 5, 10, 11]
    
    # B_B1 (bottom, left): rows 2,3 cols 0,1
    # B[2,0]=V[12], B[2,1]=V[13], B[3,0]=V[18], B[3,1]=V[19]
    B_B1 = [12, 13, 18, 19]
    
    # B_B2 (bottom, middle): rows 2,3 cols 2,3
    # B[2,2]=V[14], B[2,3]=V[15], B[3,2]=V[20], B[3,3]=V[21]
    B_B2 = [14, 15, 20, 21]
    
    # B_B3 (bottom, right): rows 2,3 cols 4,5
    # B[2,4]=V[16], B[2,5]=V[17], B[3,4]=V[22], B[3,5]=V[23]
    B_B3 = [16, 17, 22, 23]

    # C is 2x6, output matrix
    # C[i,k] corresponds to W[k*n + i] = W[k*2 + i]
    # C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10]
    # C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11]
    
    # C_L (left 2x2 block of C): [[C[0,0], C[0,1]], [C[1,0], C[1,1]]]
    # In Strassen order: c00=C[0,0]=W[0], c01=C[0,1]=W[2], c10=C[1,0]=W[1], c11=C[1,1]=W[3]
    C_L = [0, 2, 1, 3]
    
    # C_M (middle 2x2 block of C): [[C[0,2], C[0,3]], [C[1,2], C[1,3]]]
    # In Strassen order: c00=C[0,2]=W[4], c01=C[0,3]=W[6], c10=C[1,2]=W[5], c11=C[1,3]=W[7]
    C_M = [4, 6, 5, 7]
    
    # C_R (right 2x2 block of C): [[C[0,4], C[0,5]], [C[1,4], C[1,5]]]
    # In Strassen order: c00=C[0,4]=W[8], c01=C[0,5]=W[10], c10=C[1,4]=W[9], c11=C[1,5]=W[11]
    C_R = [8, 10, 9, 11]

    # Now, compute the product:
    # C = A * B
    # C_L = A_L * B_T1 + A_R * B_B1
    # C_M = A_L * B_T2 + A_R * B_B2
    # C_R = A_L * B_T3 + A_R * B_B3
    
    # We use Strassen for each of the 6 products, then add the results.
    # Total rank = 6 * 7 = 42
    
    offset = 0
    offset = add_strassen_block(offset, A_L, B_T1, C_L)  # A_L * B_T1 -> C_L
    offset = add_strassen_block(offset, A_R, B_B1, C_L)  # A_R * B_B1 -> C_L
    offset = add_strassen_block(offset, A_L, B_T2, C_M)  # A_L * B_T2 -> C_M
    offset = add_strassen_block(offset, A_R, B_B2, C_M)  # A_R * B_B2 -> C_M
    offset = add_strassen_block(offset, A_L, B_T3, C_R)  # A_L * B_T3 -> C_R
    offset = add_strassen_block(offset, A_R, B_B3, C_R)  # A_R * B_B3 -> C_R
    
    strassen_result = {"U": U, "V": V, "W": W}
    
    if verify(U, V, W):
        best_result = strassen_result
        best_rank = rank
    else:
        # Fallback to schoolbook
        best_result, best_rank = schoolbook()

    return best_result
