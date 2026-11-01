"""Matrix multiplication tensor decomposition for 2x5 by 5x5."""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm for the 2x2x2 sub-block (A11 * B11)
    # and trivial decompositions for the other blocks.
    # A is 2x5, B is 5x5, C is 2x5
    # A = [A11 (2x2)  A12 (2x3)]
    # B = [[B11 (2x2), B12 (2x3)],
    #      [B21 (3x2), B22 (3x3)]]
    # C = A * B
    # C_left (2x2) = A11*B11 + A12*B21
    # C_right (2x3) = A11*B12 + A12*B22
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 2x2x2 block (A11*B11), we need to map indices carefully.
    # A11 is rows 0-1, cols 0-1 of A
    # B11 is rows 0-1, cols 0-1 of B
    # Result goes to C[0:2, 0:2]
    
    # Let's build the full decomposition.
    # Total rank = 7 (Strassen) + 12 (trivial for A12*B21) + 12 (trivial for A11*B12) + 18 (trivial for A12*B22) = 49
    
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 49
    
    r = 0
    
    # Strassen's algorithm for A11 * B11 -> C[0:2, 0:2]
    # A11 indices in A: rows 0,1; cols 0,1 -> U indices: 0,1,2,3
    # B11 indices in B: rows 0,1; cols 0,1 -> V indices: 0,1,2,3
    # C[0:2, 0:2] indices in W: k*n+i where k in 0,1 and i in 0,1 -> W indices: 0,1,2,3
    
    # M1 = (A11+A22) * (B11+B22)
    # A11+A22: A[0,0]+A[1,1], A[0,1]+A[1,0], A[1,0]+A[0,1], A[1,1]+A[0,0]
    # Wait, A11 is 2x2: [[a00, a01], [a10, a11]]
    # A22 is 2x2: [[a11, a12], [a21, a22]] - no wait, for 2x2 Strassen:
    # A = [[A11, A12], [A21, A22]] where each is 1x1 for 2x2 case.
    # Actually for 2x2 matrices:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our case, A11 is the top-left 2x2 of A (which is 2x5)
    # A11 = A[0:2, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B11 = B[0:2, 0:2] = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C[0:2, 0:2] = A11 * B11
    
    # For Strassen on A11*B11:
    # a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1]
    # b00 = B[0,0], b01 = B[0,1], b10 = B[1,0], b11 = B[1,1]
    
    # M1 = (a00+a11)*(b00+b11)
    U[0][r] = 1.0  # a00
    U[1][r] = 1.0  # a01 - wait, M1 uses a00+a11, not a01
    # Let me redo this carefully.
    # U index for A[i,j] is i*m+j = i*5+j
    # A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[2], A[1,1] -> U[3]
    # B[0,0] -> V[0], B[0,1] -> V[1], B[1,0] -> V[2], B[1,1] -> V[3]
    # C[i,k] -> W[k*n+i] = W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0][r] = 1.0   # A[0,0]
    U[3][r] = 1.0   # A[1,1]
    V[0][r] = 1.0   # B[0,0]
    V[3][r] = 1.0   # B[1,1]
    W[0][r] = 1.0   # C[0,0]
    W[3][r] = 1.0   # C[1,1]
    r += 1
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    U[2][r] = 1.0   # A[1,0]
    U[3][r] = 1.0   # A[1,1]
    V[0][r] = 1.0   # B[0,0]
    W[2][r] = 1.0   # C[1,0]
    r += 1
    
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    U[0][r] = 1.0   # A[0,0]
    V[1][r] = 1.0   # B[0,1]
    V[3][r] = -1.0  # B[1,1]
    W[1][r] = 1.0   # C[0,1]
    r += 1
    
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    U[3][r] = 1.0   # A[1,1]
    V[2][r] = 1.0   # B[1,0]
    V[0][r] = -1.0  # B[0,0]
    W[0][r] = 1.0   # C[0,0]
    W[2][r] = 1.0   # C[1,0]
    r += 1
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    U[0][r] = 1.0   # A[0,0]
    U[1][r] = 1.0   # A[0,1]
    V[3][r] = 1.0   # B[1,1]
    W[0][r] = -1.0  # C[0,0]
    W[1][r] = 1.0   # C[0,1]
    r += 1
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2][r] = 1.0   # A[1,0]
    U[0][r] = -1.0  # A[0,0]
    V[0][r] = 1.0   # B[0,0]
    V[1][r] = 1.0   # B[0,1]
    W[3][r] = 1.0   # C[1,1]
    r += 1
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1][r] = 1.0   # A[0,1]
    U[3][r] = -1.0  # A[1,1]
    V[2][r] = 1.0   # B[1,0]
    V[3][r] = 1.0   # B[1,1]
    W[0][r] = 1.0   # C[0,0]
    r += 1
    
    # Now trivial decompositions for A12*B21 (2x3 * 3x2 -> 2x2)
    # A12 = A[0:2, 2:5], B21 = B[2:5, 0:2]
    # A12[i,k] * B21[k,j] for i in 0,1; k in 0,1,2; j in 0,1
    # A12 indices in A: A[0,2]=U[2], A[0,3]=U[3], A[0,4]=U[4], A[1,2]=U[7], A[1,3]=U[8], A[1,4]=U[9]
    # B21 indices in B: B[2,0]=V[10], B[2,1]=V[11], B[3,0]=V[15], B[3,1]=V[16], B[4,0]=V[20], B[4,1]=V[21]
    # C[0:2, 0:2] -> W[0:4]
    
    for i in range(2):
        for k in range(3):
            for j in range(2):
                U[i*5 + (2+k)][r] = 1.0
                V[(2+k)*5 + j][r] = 1.0
                W[j*2 + i][r] = 1.0
                r += 1
    
    # Trivial decompositions for A11*B12 (2x2 * 2x3 -> 2x3)
    # A11 = A[0:2, 0:2], B12 = B[0:2, 2:5]
    # A11[i,k] * B12[k,j] for i in 0,1; k in 0,1; j in 0,1,2
    # A11 indices in A: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B12 indices in B: B[0,2]=V[2], B[0,3]=V[3], B[0,4]=V[4], B[1,2]=V[7], B[1,3]=V[8], B[1,4]=V[9]
    # C[0:2, 2:5] -> W[2:5] and W[4:7]
    
    for i in range(2):
        for k in range(2):
            for j in range(3):
                U[i*5 + k][r] = 1.0
                V[k*5 + (2+j)][r] = 1.0
                W[(2+j)*2 + i][r] = 1.0
                r += 1
    
    # Trivial decompositions for A12*B22 (2x3 * 3x3 -> 2x3)
    # A12 = A[0:2, 2:5], B22 = B[2:5, 2:5]
    # A12[i,k] * B22[k,j] for i in 0,1; k in 0,1,2; j in 0,1,2
    # A12 indices in A: A[0,2]=U[2], A[0,3]=U[3], A[0,4]=U[4], A[1,2]=U[7], A[1,3]=U[8], A[1,4]=U[9]
    # B22 indices in B: B[2,2]=V[12], B[2,3]=V[13], B[2,4]=V[14], B[3,2]=V[17], B[3,3]=V[18], B[3,4]=V[19], B[4,2]=V[22], B[4,3]=V[23], B[4,4]=V[24]
    # C[0:2, 2:5] -> W[2:5] and W[4:7]
    
    for i in range(2):
        for k in range(3):
            for j in range(3):
                U[i*5 + (2+k)][r] = 1.0
                V[(2+k)*5 + (2+j)][r] = 1.0
                W[(2+j)*2 + i][r] = 1.0
                r += 1
    
    return {"U": U, "V": V, "W": W}
