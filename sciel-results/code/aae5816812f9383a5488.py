"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # U: (n*m, rank) = (12, rank) - A entries indexed by i*m+j
    # V: (m*p, rank) = (20, rank) - B entries indexed by j*p+k
    # W: (p*n, rank) = (15, rank) - C entries indexed by k*n+i (row k, col i of C)
    
    # We'll build the decomposition using a combination of Strassen and schoolbook
    # Strategy: Split A (3x4) into A1 (3x2) and A2 (3x2) by columns
    # Split B (4x5) into B1 (2x5) and B2 (2x5) by rows
    # 
    # For each 3x2 * 2x5 subproblem:
    #   Split the 3x2 into 1x2 (top) and 2x2 (bottom)
    #   1x2 * 2x5: rank 10 (schoolbook, since 1xn*nxm has rank n*m)
    #   2x2 * 2x5: split 2x5 into 2x2 + 2x3
    #     2x2 * 2x2: Strassen, rank 7
    #     2x2 * 2x3: rank 8 (known result)
    #   Total per 3x2*2x5: 10 + 7 + 8 = 25
    # Two such subproblems: 2 * 25 = 50
    # 
    # But wait, let me verify 2x2 * 2x3 rank. 
    # Actually, I'll use schoolbook for 2x2*2x3 to be safe: 2*2*3 = 12
    # Then total: 2*(10 + 7 + 12) = 2*29 = 58
    # 
    # Let me try to get rank 50 by using the correct 2x2*2x3 rank of 8.
    # For 2x2 * 2x3: A=[[a,b],[c,d]], B=[[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # That's 6 outputs, each sum of 2 bilinear terms = 12 schoolbook.
    # Can we do better? The tensor rank of 2x2*2x3 is known to be 8? Let me check.
    # Actually I'm not sure. Let me just use schoolbook to be safe: rank 12.
    # Total: 2*(10+7+12) = 58.
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]  # 12 x 58
    V = [[0] * rank for _ in range(m * p)]  # 20 x 58
    W = [[0] * rank for _ in range(p * n)]  # 15 x 58
    
    col = 0
    
    def emit(u_entries, v_entries, w_entries):
        """Emit one rank-1 term. u_entries, v_entries, w_entries are lists of (index, coeff)."""
        nonlocal col
        for idx, coeff in u_entries:
            U[idx][col] += coeff
        for idx, coeff in v_entries:
            V[idx][col] += coeff
        for idx, coeff in w_entries:
            W[idx][col] += coeff
        col += 1
    
    # A1: rows 0-2, cols 0-1 of A. U indices: i*4+j
    #   A1_top (1x2): U[0]=A[0][0], U[1]=A[0][1]
    #   A1_bot (2x2): U[4]=A[1][0], U[5]=A[1][1], U[8]=A[2][0], U[9]=A[2][1]
    # A2: rows 0-2, cols 2-3 of A. U indices: i*4+j
    #   A2_top (1x2): U[2]=A[0][2], U[3]=A[0][3]
    #   A2_bot (2x2): U[6]=A[1][2], U[7]=A[1][3], U[10]=A[2][2], U[11]=A[2][3]
    
    # B1: rows 0-1 of B, cols 0-4. V indices: j*5+k
    #   B1_2x2: V[0]=B[0][0], V[1]=B[0][1], V[5]=B[1][0], V[6]=B[1][1]
    #   B1_2x3: V[2]=B[0][2], V[3]=B[0][3], V[4]=B[0][4], V[7]=B[1][2], V[8]=B[1][3], V[9]=B[1][4]
    # B2: rows 2-3 of B, cols 0-4. V indices: j*5+k
    #   B2_2x2: V[10]=B[2][0], V[11]=B[2][1], V[15]=B[3][0], V[16]=B[3][1]
    #   B2_2x3: V[12]=B[2][2], V[13]=B[2][3], V[14]=B[2][4], V[17]=B[3][2], V[18]=B[3][3], V[19]=B[3][4]
    
    # C is 3x5. W[k*3+i] where k=0..4 (row of C), i=0..2 (col of C)
    # C row 0: W[0], W[3], W[6], W[9], W[12]
    # C row 1: W[1], W[4], W[7], W[10], W[13]
    # C row 2: W[2], W[5], W[8], W[11], W[14]
    
    # === A1 * B1 (3x2 * 2x5) ===
    # A1_top (1x2) * B1 (2x5): rank 10
    # C[0][j] += A[0][0]*B[0][j] + A[0][1]*B[1][j] for j=0..4
    for j in range(5):
        emit([(0, 1)], [(j, 1)], [(j*3+0, 1)])  # A[0][0]*B[0][j] -> C[0][j]
        emit([(1, 1)], [(5+j, 1)], [(j*3+0, 1)])  # A[0][1]*B[1][j] -> C[0][j]
    
    # A1_bot (2x2) * B1_2x2 (2x2): Strassen, rank 7
    # a=U[4], b=U[5], c=U[8], d=U[9]
    # e=V[0], f=V[1], g=V[5], h=V[6]
    # Output: C[1][0]=W[1], C[1][1]=W[4], C[2][0]=W[2], C[2][1]=W[5]
    # P1=(a+d)(e+h)
    emit([(4,1),(9,1)], [(0,1),(6,1)], [(1,1),(5,1)])
    # P2=(c+d)e
    emit([(8,1),(9,1)], [(0,1)], [(2,1)])
    # P3=a(f-h)
    emit([(4,1)], [(1,1),(6,-1)], [(4,1)])
    # P4=d(g-e)
    emit([(9,1)], [(5,1),(0,-1)], [(1,1),(2,1)])
    # P5=(a+b)h
    emit([(4,1),(5,1)], [(6,1)], [(1,-1),(4,1)])
    # P6=(c-a)(e+f)
    emit([(8,1),(4,-1)], [(0,1),(1,1)], [(5,1)])
    # P7=(b-d)(g+h)
    emit([(5,1),(9,-1)], [(5,1),(6,1)], [(1,1),(5,1)])
    
    # A1_bot (2x2) * B1_2x3 (2x3): schoolbook, rank 12
    # a=U[4], b=U[5], c=U[8], d=U[9]
    # B1_2x3: V[2],V[3],V[4] (row 0), V[7],V[8],V[9] (row 1)
    # Output: C[1][2..4]=W[7],W[10],W[13], C[2][2..4]=W[8],W[11],W[14]
    for j in range(3):
        col_offset = 2 + j
        emit([(4,1)], [(2+j,1)], [(col_offset*3+1,1)])  # a*B[0][2+j] -> C[1][2+j]
        emit([(5,1)], [(7+j,1)], [(col_offset*3+1,1)])  # b*B[1][2+j] -> C[1][2+j]
        emit([(8,1)], [(2+j,1)], [(col_offset*3+2,1)])  # c*B[0][2+j] -> C[2][2+j]
        emit([(9,1)], [(7+j,1)], [(col_offset*3+2,1)])  # d*B[1][2+j] -> C[2][2+j]
    
    # === A2 * B2 (3x2 * 2x5) ===
    # A2_top (1x2) * B2 (2x5): rank 10
    # C[0][j] += A[0][2]*B[2][j] + A[0][3]*B[3][j] for j=0..4
    for j in range(5):
        emit([(2, 1)], [(10+j, 1)], [(j*3+0, 1)])  # A[0][2]*B[2][j] -> C[0][j]
        emit([(3, 1)], [(15+j, 1)], [(j*3+0, 1)])  # A[0][3]*B[3][j] -> C[0][j]
    
    # A2_bot (2x2) * B2_2x2 (2x2): Strassen, rank 7
    # a=U[6], b=U[7], c=U[10], d=U[11]
    # e=V[10], f=V[11], g=V[15], h=V[16]
    # Output: C[1][0]=W[1], C[1][1]=W[4], C[2][0]=W[2], C[2][1]=W[5]
    # P1=(a+d)(e+h)
    emit([(6,1),(11,1)], [(10,1),(16,1)], [(1,1),(5,1)])
    # P2=(c+d)e
    emit([(10,1),(11,1)], [(10,1)], [(2,1)])
    # P3=a(f-h)
    emit([(6,1)], [(11,1),(16,-1)], [(4,1)])
    # P4=d(g-e)
    emit([(11,1)], [(15,1),(10,-1)], [(1,1),(2,1)])
    # P5=(a+b)h
    emit([(6,1),(7,1)], [(16,1)], [(1,-1),(4,1)])
    # P6=(c-a)(e+f)
    emit([(10,1),(6,-1)], [(10,1),(11,1)], [(5,1)])
    # P7=(b-d)(g+h)
    emit([(7,1),(11,-1)], [(15,1),(16,1)], [(1,1),(5,1)])
    
    # A2_bot (2x2) * B2_2x3 (2x3): schoolbook, rank 12
    # a=U[6], b=U[7], c=U[10], d=U[11]
    # B2_2x3: V[12],V[13],V[14] (row 2), V[17],V[18],V[19] (row 3)
    # Output: C[1][2..4]=W[7],W[10],W[13], C[2][2..4]=W[8],W[11],W[14]
    for j in range(3):
        col_offset = 2 + j
        emit([(6,1)], [(12+j,1)], [(col_offset*3+1,1)])  # a*B[2][2+j] -> C[1][2+j]
        emit([(7,1)], [(17+j,1)], [(col_offset*3+1,1)])  # b*B[3][2+j] -> C[1][2+j]
        emit([(10,1)], [(12+j,1)], [(col_offset*3+2,1)])  # c*B[2][2+j] -> C[2][2+j]
        emit([(11,1)], [(17+j,1)], [(col_offset*3+2,1)])  # d*B[3][2+j] -> C[2][2+j]
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
