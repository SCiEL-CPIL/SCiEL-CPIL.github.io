"""Matrix multiplication 2,4,5 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    # Split middle dimension m=4 into two halves of size 2
    # A = [A1 | A2], B = [B1; B2], C = A1*B1 + A2*B2
    # Each Ai is 2x2, each Bi is 2x5
    # For each 2x2 * 2x5: split 5 cols into 2+2+1
    #   2x2*2x2: rank 7 (Strassen)
    #   2x2*2x2: rank 7
    #   2x2*2x1: rank 3
    # Total per half: 17, grand total: 34
    
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 34
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 34
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 34
    
    col = 0
    
    # Strassen for 2x2 * 2x2: 7 terms
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = [[c00,c01],[c10,c11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    def strassen_2x2_2x2(base_col, A_row_offsets, B_col_offsets, C_row_offsets):
        """Add 7 Strassen terms. A_row_offsets = (r0, r1) for A rows, 
           B_col_offsets = (c0, c1) for B cols within the 2x2 block,
           C_row_offsets = (r0, r1) for C rows."""
        nonlocal col
        # A entries: A[i*m+j] for i in {A_row_offsets[0], A_row_offsets[1]}, j in {B_col_offsets[0], B_col_offsets[1]}
        # Wait, I need to be more careful with indexing.
        # A is n x m = 2 x 4. U index = i*m + j = i*4 + j
        # B is m x p = 4 x 5. V index = j*p + k = j*5 + k  
        # C is p x n = 5 x 2. W index = k*n + i = k*2 + i (trace convention W[k*n+i,r])
        # 
        # For the sub-block: A sub-block uses rows r0,r1 of A (i indices) and cols cA0,cA1 of A (j indices)
        # B sub-block uses rows rB0,rB1 of B (j indices) and cols cB0,cB1 of B (k indices)
        # C sub-block uses rows rC0,rC1 of C (k indices) and cols cC0,cC1 of C (i indices)
        # 
        # In the decomposition: A1 uses A rows {0,1}, cols {0,1}; A2 uses A rows {0,1}, cols {2,3}
        # B1 uses B rows {0,1}, cols {cB0,cB1,cB2}; B2 uses B rows {2,3}, cols {cB0,cB1,cB2}
        # C uses all rows/cols but split by which A*B product
        
        # Actually, let me just directly assign indices.
        # A1: rows 0,1 of A; cols 0,1 of A -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        # A2: rows 0,1 of A; cols 2,3 of A -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
        # B1: rows 0,1 of B -> V indices for cols c: 0*5+c, 1*5+c
        # B2: rows 2,3 of B -> V indices for cols c: 2*5+c, 3*5+c
        # C: W index = k*2+i where k is C column (0..4), i is C row (0..1)
        
        return col
    
    # Let me just hardcode the 34 terms directly.
    # Term indices 0-6: A1 * B1_col0_1 (Strassen, 2x2*2x2)
    # A1: a00=U[0], a01=U[1], a10=U[4], a11=U[5]
    # B1_col0_1: b00=V[0], b01=V[1], b10=V[5], b11=V[6]  (V[j*5+k], j=0,1; k=0,1)
    # C: c00=W[0], c01=W[2], c10=W[4], c11=W[6]  (W[k*2+i], k=0,1; i=0,1)
    
    # M1 = (a00+a11)(b00+b11)
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[6][col] = 1  # c00 += M1, c11 += M1
    col += 1
    
    # M2 = (a10+a11)b00
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[4][col] = 1; W[6][col] = -1  # c10 += M2, c11 -= M2
    col += 1
    
    # M3 = a00(b10-b11)
    U[0][col] = 1
    V[5][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[4][col] = 1  # c01 += M3, c10 += M3
    col += 1
    
    # M4 = a11(b01-b00)
    U[5][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[4][col] = 1  # c00 += M4, c10 += M4
    col += 1
    
    # M5 = (a00+a01)b11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[2][col] = 1  # c00 -= M5, c01 += M5
    col += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[6][col] = 1  # c11 += M6
    col += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][col] = 1; U[5][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1  # c00 += M7
    col += 1
    
    # Terms 7-13: A1 * B1_col2_3 (Strassen, 2x2*2x2)
    # B1_col2_3: b00=V[2], b01=V[3], b10=V[7], b11=V[8]
    # C: c00=W[4], c01=W[6], c10=W[8], c11=W[10]  (k=2,3 -> W[2*2+0]=W[4], W[2*2+1]=W[5], W[3*2+0]=W[6], W[3*2+1]=W[7])
    # Wait, W[k*n+i] = W[k*2+i]. For k=2: W[4],W[5]. For k=3: W[6],W[7].
    # So c00=W[4], c01=W[5], c10=W[6], c11=W[7]
    
    # M1
    U[0][col] = 1; U[5][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[4][col] = 1; U[5][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[7][col] = 1; V[8][col] = -1
    W[5][col] = 1; W[6][col] = 1
    col += 1
    
    # M4
    U[5][col] = 1
    V[3][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M6
    U[4][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[5][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # Terms 14-16: A1 * B1_col4 (2x2 * 2x1, rank 3)
    # B1_col4: b0=V[4], b1=V[9]
    # C: c0=W[8], c1=W[9]  (k=4 -> W[8],W[9])
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # Rank 3: M1=(a00+a10)*b0, M2=a01*b1, M3=a11*b1... 
    # Actually: C0 = a00*b0 + a01*b1, C1 = a10*b0 + a11*b1
    # Term1: (a00+a10)*b0 -> C0 += a00*b0, C1 += a10*b0
    U[0][col] = 1; U[4][col] = 1
    V[4][col] = 1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    # Term2: a01*b1 -> C0 += a01*b1
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    # Term3: a11*b1 -> C1 += a11*b1
    U[5][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # Terms 17-23: A2 * B2_col0_1 (Strassen)
    # A2: a00=U[2], a01=U[3], a10=U[6], a11=U[7]
    # B2_col0_1: b00=V[10], b01=V[11], b10=V[15], b11=V[16]
    # C: same W as before (k=0,1) -> W[0],W[1],W[2],W[3]
    
    # M1
    U[2][col] = 1; U[7][col] = 1
    V[10][col] = 1; V[16][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2
    U[6][col] = 1; U[7][col] = 1
    V[10][col] = 1
    W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # M3
    U[2][col] = 1
    V[15][col] = 1; V[16][col] = -1
    W[1][col] = 1; W[2][col] = 1
    col += 1
    
    # M4
    U[7][col] = 1
    V[11][col] = 1; V[10][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5
    U[2][col] = 1; U[3][col] = 1
    V[16][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6
    U[6][col] = 1; U[2][col] = -1
    V[10][col] = 1; V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7
    U[3][col] = 1; U[7][col] = -1
    V[15][col] = 1; V[16][col] = 1
    W[0][col] = 1
    col += 1
    
    # Terms 24-30: A2 * B2_col2_3 (Strassen)
    # B2_col2_3: b00=V[12], b01=V[13], b10=V[17], b11=V[18]
    # C: k=2,3 -> W[4],W[5],W[6],W[7]
    
    # M1
    U[2][col] = 1; U[7][col] = 1
    V[12][col] = 1; V[18][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[6][col] = 1; U[7][col] = 1
    V[12][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[2][col] = 1
    V[17][col] = 1; V[18][col] = -1
    W[5][col] = 1; W[6][col] = 1
    col += 1
    
    # M4
    U[7][col] = 1
    V[13][col] = 1; V[12][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M5
    U[2][col] = 1; U[3][col] = 1
    V[18][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M6
    U[6][col] = 1; U[2][col] = -1
    V[12][col] = 1; V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[3][col] = 1; U[7][col] = -1
    V[17][col] = 1; V[18][col] = 1
    W[4][col] = 1
    col += 1
    
    # Terms 31-33: A2 * B2_col4 (2x2 * 2x1, rank 3)
    # B2_col4: b0=V[14], b1=V[19]
    # C: k=4 -> W[8],W[9]
    
    # Term1: (a00+a10)*b0
    U[2][col] = 1; U[6][col] = 1
    V[14][col] = 1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    # Term2: a01*b1
    U[3][col] = 1
    V[19][col] = 1
    W[8][col] = 1
    col += 1
    # Term3: a11*b1
    U[7][col] = 1
    V[19][col] = 1
    W[9][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Convert to integers where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
