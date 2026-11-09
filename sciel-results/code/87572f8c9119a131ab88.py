"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Reconstruct: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # U indices: i*m+j where i=0,1 and j=0,1
    # V indices: j*p+k where j=0,1 and k=0,1
    # W indices: k*n+i where k=0,1 and i=0,1
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0  # a11
    U[3][0] = 1.0  # a22
    V[0][0] = 1.0  # b11
    V[3][0] = 1.0  # b22
    W[0][0] = 1.0  # c11 (k=0,i=0)
    W[3][0] = 1.0  # c22 (k=1,i=1)
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0  # a21
    U[3][1] = 1.0  # a22
    V[0][1] = 1.0  # b11
    W[2][1] = 1.0  # c21 (k=0,i=1)
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0  # b12
    V[3][2] = -1.0 # b22
    W[1][2] = 1.0  # c12 (k=1,i=0)
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0  # b21
    V[0][3] = -1.0 # b11
    W[0][3] = 1.0  # c11 (k=0,i=0)
    W[2][3] = 1.0  # c21 (k=0,i=1)
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0  # a11
    U[1][4] = 1.0  # a12
    V[3][4] = 1.0  # b22
    W[0][4] = -1.0 # c11 (k=0,i=0)
    W[1][4] = 1.0  # c12 (k=1,i=0)
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0  # a21
    U[0][5] = -1.0 # a11
    V[0][5] = 1.0  # b11
    V[1][5] = 1.0  # b12
    W[3][5] = 1.0  # c22 (k=1,i=1)
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0  # a12
    U[3][6] = -1.0 # a22
    V[2][6] = 1.0  # b21
    V[3][6] = 1.0  # b22
    W[0][6] = 1.0  # c11 (k=0,i=0)
    W[3][6] = 1.0  # c22 (k=1,i=1)
    
    return U, V, W

def _combine_decompositions(decs):
    """Combine multiple decompositions by concatenation."""
    total_rank = sum(len(d[0][0]) for d in decs)
    n = m = p = 2
    U = [[0.0] * total_rank for _ in range(n*m)]
    V = [[0.0] * total_rank for _ in range(m*p)]
    W = [[0.0] * total_rank for _ in range(p*n)]
    
    offset = 0
    for U_d, V_d, W_d in decs:
        r = len(U_d[0])
        for i in range(n*m):
            for rr in range(r):
                U[i][offset+rr] += U_d[i][rr]
        for i in range(m*p):
            for rr in range(r):
                V[i][offset+rr] += V_d[i][rr]
        for i in range(p*n):
            for rr in range(r):
                W[i][offset+rr] += W_d[i][rr]
        offset += r
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using block methods."""
    n, m, p = 2, 4, 7
    
    # Strategy: Use a combination of Strassen and schoolbook for different blocks.
    # We'll split the 4x7 matrix B into blocks and use Strassen where possible.
    
    # Split B into two parts: B1 (4x4) and B2 (4x3)
    # A@B = A@B1 + A@B2
    # A@B1 is 2x4 times 4x4 -> 2x4
    # A@B2 is 2x4 times 4x3 -> 2x3
    
    # For A@B1 (2x4 x 4x4):
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B1 into B1_top (2x4) and B1_bottom (2x4)
    # A@B1 = A1@B1_top + A2@B1_bottom
    # Each is 2x2 times 2x4 -> 2x4
    # For M_{2,2,4}, we can use schoolbook (rank 8) or try Strassen-like.
    # Actually, for 2x2x4, we can split the 4 columns into two 2-column blocks.
    # Each 2x2x2 can be done with Strassen (rank 7).
    # So M_{2,2,4} can be done in rank 14 (two Strassen blocks).
    # But schoolbook is rank 8, which is better!
    # So use schoolbook for M_{2,2,4}: rank 8 each, total 16 for A@B1.
    
    # For A@B2 (2x4 x 4x3):
    # Similarly: A1@B2_top + A2@B2_bottom, each M_{2,2,3}
    # M_{2,2,3}: schoolbook rank 6. No known improvement.
    # Total for A@B2: 12
    
    # Grand total: 16 + 12 = 28
    
    # But wait, this is just schoolbook split, which gives the same total as original schoolbook (56).
    # Actually no: original schoolbook is 2*4*7 = 56.
    # This split gives 2*8 + 2*6 = 28. That's half!
    
    # Wait, I think I made an error. Let me recalculate.
    # M_{2,2,4} schoolbook: 2*2*4 = 16, not 8.
    # M_{2,2,3} schoolbook: 2*2*3 = 12, not 6.
    # So total: 2*16 + 2*12 = 56. Same as before.
    
    # So this split doesn't help. We need a better approach.
    
    # Let me try a different strategy: use Strassen on the 4x4 part.
    # For A@B1 (2x4 x 4x4):
    # View this as multiplying a 2x4 matrix by a 4x4 matrix.
    # We can think of this as 4 separate 2x1 x 1x4 multiplications, but that's schoolbook.
    # Alternatively, we can use the fact that 4x4 multiplication can be done with Strassen.
    # But we're multiplying 2x4 by 4x4, not 4x4 by 4x4.
    
    # Let me try a direct construction with rank 35.
    
    # Actually, let's use a known result: the rank of M_{2,4,7} is at most 35.
    # This can be achieved using a combination of Strassen and schoolbook.
    
    # For now, let's return a safe schoolbook decomposition with rank 56.
    # In a future iteration, we can search for better decompositions.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a low-rank decomposition
    U, V, W = _build_2x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
    
    # Convert to lists if needed
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
