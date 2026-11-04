"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # We'll build the decomposition hierarchically.
    
    def strassen_222():
        """Returns U,V,W for 2x2 x 2x2 multiplication, rank 7.
        U: 4x7 (i*2+j, term)
        V: 4x7 (j*2+k, term)
        W: 4x7 (k*2+i, term)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (A00+A11)*(B00+B11)
        # C00 += 1, C11 += 1
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # Term 1: M2 = (A10+A11)*B00
        # C10 += 1, C11 -= 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # Term 2: M3 = A00*(B01-B11)
        # C01 += 1, C11 += 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # Term 3: M4 = A11*(B10-B00)
        # C10 += 1, C11 += 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[3][3] = 1
        
        # Term 4: M5 = (A00+A01)*B11
        # C00 -= 1, C01 += 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # Term 5: M6 = (A10-A00)*(B00+B10)
        # C11 += 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # Term 6: M7 = (A01-A11)*(B01+B11)
        # C00 += 1, C11 += 1
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W
    
    def dot_122():
        """Returns U,V,W for 1x2 x 2x2 multiplication, rank 4.
        U: 2x4 (j, term)  [i is always 0, so U has m=2 rows]
        V: 4x4 (j*2+k, term)
        W: 4x4 (k*1+i, term) = (k, term) since n=1
        """
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # 4 terms: (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(2)]
        
        for t, (j, k) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            U[j][t] = 1
            V[j*2+k][t] = 1
            W[k*1+0][t] = 1  # W[k*n+i] = W[k*1+0] = W[k]
        
        return U, V, W
    
    def mul_322():
        """Returns U,V,W for 3x2 x 2x2 multiplication, rank 11.
        U: 6x11 (i*2+j, term)
        V: 4x11 (j*2+k, term)
        W: 4x11 (k*3+i, term)
        """
        # C = A_top (2x2) * B (2x2) stacked on A_bot (1x2) * B (2x2)
        # Top part: Strassen 2x2x2, rank 7
        # Bot part: 1x2x2, rank 4
        U_str, V_str, W_str = strassen_222()
        U_dot, V_dot, W_dot = dot_122()
        
        rank = 7 + 4
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(6)]  # p*n = 2*3 = 6
        
        # Top: i=0,1; U rows 0..3, W rows 0..3 (k*2+i for n=2)
        # But in 3x2x2, W index is k*3+i
        # For top part (i=0,1): W[k*3+0] and W[k*3+1]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2+j][r] = U_str[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][r] = V_str[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[k*3+i][r] = W_str[k*2+i][r]
        
        # Bottom: i=2; U rows 4,5 (i=2, j=0,1)
        # W for bottom: k*3+2
        for r in range(4):
            for j in range(2):
                U[2*2+j][7+r] = U_dot[j][r]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][7+r] = V_dot[j*2+k][r]
            for k in range(2):
                W[k*3+2][7+r] = W_dot[k][r]
        
        return U, V, W
    
    def mul_342():
        """Returns U,V,W for 3x4 x 4x2 multiplication, rank 22.
        U: 12x22 (i*4+j, term)
        V: 8x22 (j*2+k, term)
        W: 6x22 (k*3+i, term)
        """
        # Split A (3x4) into A1 (3x2) and A2 (3x2)
        # Split B (4x2) into B1 (2x2) and B2 (2x2)
        # C = A1*B1 + A2*B2
        # Each 3x2 x 2x2 has rank 11
        U1, V1, W1 = mul_322()
        U2, V2, W2 = mul_322()
        
        rank = 22
        U = [[0]*rank for _ in range(12)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(6)]
        
        # First half: A1 (cols 0,1) * B1 (rows 0,1)
        for r in range(11):
            for i in range(3):
                for j in range(2):
                    U[i*4+j][r] = U1[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][r] = V1[j*2+k][r]
            for k in range(2):
                for i in range(3):
                    W[k*3+i][r] = W1[k*3+i][r]
        
        # Second half: A2 (cols 2,3) * B2 (rows 2,3)
        for r in range(11):
            for i in range(3):
                for j in range(2):
                    U[i*4+(2+j)][11+r] = U2[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*2+k][11+r] = V2[j*2+k][r]
            for k in range(2):
                for i in range(3):
                    W[k*3+i][11+r] = W2[k*3+i][r]
        
        return U, V, W
    
    def mul_348():
        """Returns U,V,W for 3x4 x 4x8 multiplication, rank 88.
        U: 12x88 (i*4+j, term)
        V: 32x88 (j*4+k, term)
        W: 24x88 (k*3+i, term)
        """
        # Split B (4x8) into 4 blocks of 4x2
        # C = A*B1 + A*B2 + A*B3 + A*B4
        # Each 3x4 x 4x2 has rank 22
        rank = 88
        U = [[0]*rank for _ in range(12)]
        V = [[0]*rank for _ in range(32)]
        W = [[0]*rank for _ in range(24)]
        
        for block in range(4):
            U1, V1, W1 = mul_342()
            base = block * 22
            for r in range(22):
                for i in range(3):
                    for j in range(4):
                        U[i*4+j][base+r] = U1[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V[j*4+(block*2+k)][base+r] = V1[j*2+k][r]
                for k in range(2):
                    for i in range(3):
                        W[(block*2+k)*3+i][base+r] = W1[k*3+i][r]
        
        return U, V, W
    
    return mul_348()
# EVOLVE-BLOCK-END
