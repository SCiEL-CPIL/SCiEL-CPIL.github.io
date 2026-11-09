"""Matrix multiplication 2x3x3 with Strassen-inspired block decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3  # A is 2x3, B is 3x3, C is 2x3
    
    # We model this as:
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1
    # B = [[B11 B12], [B21 B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
    # C = [C1 C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    #
    # C1 = A1*B11 + A2*B21 (2x2 result)
    # C2 = A1*B12 + A2*B22 (2x1 result)
    #
    # For C1: A1 (2x2) * B11 (2x2) + A2 (2x1) * B21 (1x2)
    #   The first term is 2x2x2 matmul: use Strassen rank 7
    #   The second term is 2x1x1x2 = essentially 4 scalar products: rank 4
    #   Total for C1: 7 + 4 = 11
    #
    # For C2: A1 (2x2) * B12 (2x1) + A2 (2x1) * B22 (1x1)
    #   The first term is 2x2x2x1: this is like multiplying 2x2 by 2x1, which needs 4 mults
    #   The second term is 2x1x1x1: 2 scalar mults
    #   Total for C2: 4 + 2 = 6
    #
    # Total rank: 11 + 6 = 17
    #
    # Let's construct this explicitly.
    #
    # Strassen for 2x2:
    # Given M = [[m11, m12], [m21, m22]], N = [[n11, n12], [n21, n22]]
    # P1 = (m11+m21)*(n11+n21)
    # P2 = (m11+m21)*n22
    # P3 = m11*(n11-n21)
    # P4 = m22*(n12-n11)
    # P5 = (m12+m22)*(n11+n21)
    # P6 = (m11-m21)*(n21+n22)
    # P7 = (m12-m22)*(n11+n12)
    # C11 = P1+P4-P5+P7
    # C12 = P2+P5
    # C21 = P3+P6
    # C22 = P1-P2+P3+P4
    #
    # Let's label A1 = [[a00, a01], [a10, a11]] (rows 0,1 of A; cols 0,1)
    # A2 = [a02, a12] (col 2 of A)
    # B11 = [[b00, b01], [b10, b11]] (rows 0,1 of B; cols 0,1)
    # B12 = [b02, b12] (rows 0,1 of B; col 2)
    # B21 = [b20, b21] (row 2 of B; cols 0,1)
    # B22 = b22 (row 2 of B; col 2)
    #
    # C1 (2x2):
    # C1[0,0] = a00*b00 + a01*b10 + a02*b20 + a10*0 + a11*0 + a12*0 = wait no
    # C1[0,0] = a00*b00 + a01*b10 + a02*b20
    # C1[0,1] = a00*b01 + a01*b11 + a02*b21
    # C1[1,0] = a10*b00 + a11*b10 + a12*b20
    # C1[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # So C1 = A1*B11 + A2*B21 where A2*B21 is outer product:
    # A2*B21 = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    #
    # C2 (2x1):
    # C2[0,0] = a00*b02 + a01*b12 + a02*b22
    # C2[1,0] = a10*b02 + a11*b12 + a12*b22
    
    # Let's build the decomposition with rank 17.
    # Variables: a00,a01,a02,a10,a11,a12 (A entries)
    #            b00,b01,b02,b10,b11,b12,b20,b21,b22 (B entries)
    # 
    # U matrix: rows indexed by (i,j) for A, so 6 rows
    # V matrix: rows indexed by (j,k) for B, so 9 rows
    # W matrix: rows indexed by (k,n+i) for C, so 6 rows (using W[k*n+i] convention)
    #   C is 2x3, so C[i,k] for i=0,1 and k=0,1,2
    #   W index: k*n + i = k*2 + i
    
    rank = 17
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # A indices: A[i,j] -> row i*m+j = i*3+j
    # B indices: B[j,k] -> row j*p+k = j*3+k
    # C indices: C[i,k] -> W index k*n+i = k*2+i
    
    # Strassen for A1*B11 (2x2x2), 7 terms
    # P1 = (a00+a10)*(b00+b10)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a10
    V[0][0] = 1  # b00
    V[3][0] = 1  # b10
    # contributes to C1[0,0], C1[1,0], C1[1,1] with +1
    # C1[0,0] -> W[0*2+0]=W[0]
    # C1[1,0] -> W[0*2+1]=W[1]
    # C1[1,1] -> W[1*2+1]=W[3]
    W[0][0] = 1
    W[1][0] = 1
    W[3][0] = 1
    
    # P2 = (a00+a10)*b11
    U[0][1] = 1
    U[3][1] = 1
    V[3][1] = 1  # b11
    # C1[0,1] -> W[1*2+0]=W[2]
    # C1[1,1] -> W[3]
    W[2][1] = 1
    W[3][1] = 1
    
    # P3 = a00*(b10-b00)
    U[0][2] = 1
    V[0][2] = -1  # -b00
    V[3][2] = 1   # +b10
    # C1[1,0] -> W[1]
    # C1[1,1] -> W[3]
    W[1][2] = 1
    W[3][2] = 1
    
    # P4 = a11*(b01-b00)
    U[4][3] = 1  # a11 (row index for i=1,j=1 is 1*3+1=4)
    V[0][3] = -1  # -b00
    V[1][3] = 1   # +b01
    # C1[0,0] -> W[0]
    # C1[1,1] -> W[3]
    W[0][3] = 1
    W[3][3] = 1
    
    # P5 = (a01+a11)*(b00+b10)
    U[1][4] = 1  # a01 (row 0*3+1=1)
    U[4][4] = 1  # a11
    V[0][4] = 1  # b00
    V[3][4] = 1  # b10
    # C1[0,1] -> W[2]
    # C1[1,1] -> W[3]
    W[2][4] = -1
    W[3][4] = -1
    
    # P6 = (a00-a10)*(b10+b11)
    U[0][5] = 1
    U[3][5] = -1
    V[3][5] = 1  # b10
    V[4][5] = 1  # b11 (row 1*3+1=4)
    # C1[1,0] -> W[1]
    W[1][5] = 1
    
    # P7 = (a01-a11)*(b00+b01)
    U[1][6] = 1
    U[4][6] = -1
    V[0][6] = 1  # b00
    V[1][6] = 1  # b01
    # C1[0,0] -> W[0]
    W[0][6] = 1
    
    # Now the cross terms A2*B21 (4 terms)
    # a02*b20 -> C1[0,0] -> W[0]
    U[2][7] = 1  # a02 (row 0*3+2=2)
    V[6][7] = 1  # b20 (row 2*3+0=6)
    W[0][7] = 1
    
    # a02*b21 -> C1[0,1] -> W[2]
    U[2][8] = 1
    V[7][8] = 1  # b21 (row 2*3+1=7)
    W[2][8] = 1
    
    # a12*b20 -> C1[1,0] -> W[1]
    U[5][9] = 1  # a12 (row 1*3+2=5)
    V[6][9] = 1
    W[1][9] = 1
    
    # a12*b21 -> C1[1,1] -> W[3]
    U[5][10] = 1
    V[7][10] = 1
    W[3][10] = 1
    
    # Now C2 terms: A1*B12 (4 terms) + A2*B22 (2 terms)
    # a00*b02 -> C2[0,0] -> W[2*2+0]=W[4]
    U[0][11] = 1
    V[2][11] = 1  # b02 (row 0*3+2=2)
    W[4][11] = 1
    
    # a01*b12 -> C2[0,0] -> W[4]
    U[1][12] = 1
    V[5][12] = 1  # b12 (row 1*3+2=5)
    W[4][12] = 1
    
    # a10*b02 -> C2[1,0] -> W[2*2+1]=W[5]
    U[3][13] = 1
    V[2][13] = 1
    W[5][13] = 1
    
    # a11*b12 -> C2[1,0] -> W[5]
    U[4][14] = 1
    V[5][14] = 1
    W[5][14] = 1
    
    # a02*b22 -> C2[0,0] -> W[4]
    U[2][15] = 1
    V[8][15] = 1  # b22 (row 2*3+2=8)
    W[4][15] = 1
    
    # a12*b22 -> C2[1,0] -> W[5]
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
