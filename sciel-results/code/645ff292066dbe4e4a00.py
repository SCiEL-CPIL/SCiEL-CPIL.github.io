import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # Rank 7
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0..3 for 2x2 blocks (00, 01, 10, 11)
    
    U2 = [
        [1, 0, 1, 0, 1, 1, 0], # 00
        [0, 0, 0, 1, 1, 0, 1], # 01
        [1, 1, 0, 0, 0, -1, 0],# 10
        [0, 1, 1, 1, 0, 0, -1] # 11
    ]
    
    V2 = [
        [1, 1, 0, 0, 0, 0, 1], # 00
        [0, 0, 1, 1, 0, 0, 0], # 01
        [1, 0, 0, 0, 1, 1, 0], # 10
        [0, 1, 1, -1, 0, 0, 1] # 11
    ]
    
    W2 = [
        [1, 0, 0, 1, 0, 0, 1], # 00
        [0, 1, 1, 0, 0, 0, 0], # 01
        [0, 0, 1, 0, 1, 1, 0], # 10
        [1, 1, 0, -1, 0, 0, 1] # 11
    ]
    
    # Verify 2x2 Strassen
    # C = A * B
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My U2, V2, W2 above are just guesses. I need to ensure they are correct.
    # Let's construct them properly from the formulas.
    
    # U2[i, k] is the coefficient of A_i in M_k
    # V2[j, k] is the coefficient of B_j in M_k
    # W2[r, k] is the coefficient of M_k in C_r
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # k=0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # k=1 (M2)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # k=2 (M3)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # k=3 (M4)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # k=4 (M5)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # k=5 (M6)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # k=6 (M7)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now form 4x4 factors using Kronecker product
    # The 4x4 matrix A is composed of 2x2 blocks A00, A01, A10, A11
    # Indexing: A[i, j] where i,j in 0..3
    # Block index b = i//2 * 2 + j//2? No.
    # A is 4x4.
    # A[0:2, 0:2] is A00.
    # A[0:2, 2:4] is A01.
    # A[2:4, 0:2] is A10.
    # A[2:4, 2:4] is A11.
    
    # The tensor for 4x4 is M4 = M2 (A) * M2 (B) * M2 (C) in some sense?
    # No, M4 = M2 \otimes M2.
    # The factors are U4 = U2 \otimes U2, etc.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to list of lists as required
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
