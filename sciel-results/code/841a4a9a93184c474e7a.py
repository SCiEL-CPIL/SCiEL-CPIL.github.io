import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 3x3 Schoolbook (Rank 27)
    # Indices i, j, k in 0..2
    # U3 row: i*m + j -> i*3 + j
    # V3 row: j*p + k -> j*3 + k
    # W3 row: k*n + i -> k*3 + i
    
    rank3 = 27
    U3 = np.zeros((9, rank3))
    V3 = np.zeros((9, rank3))
    W3 = np.zeros((9, rank3))
    
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j, c] = 1
                V3[j*3 + k, c] = 1
                W3[k*3 + i, c] = 1
                c += 1
                
    # 2x2 Strassen (Rank 7)
    # Indices alpha, beta, gamma in 0..1
    # U2 row: alpha*2 + beta
    # V2 row: beta*2 + gamma
    # W2 row: gamma*2 + alpha
    
    rank2 = 7
    U2 = np.zeros((4, rank2))
    V2 = np.zeros((4, rank2))
    W2 = np.zeros((4, rank2))
    
    # Strassen terms
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1 # Wait, W is output.
    # Let's use the standard Strassen formulation carefully.
    
    # Reset U2, V2, W2 to 0
    U2 = np.zeros((4, rank2))
    V2 = np.zeros((4, rank2))
    W2 = np.zeros((4, rank2))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00 and C11 get M1?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define W2 based on the output contributions.
    # W2[row, col] is 1 if term col contributes to output row.
    
    # Term 0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1 (M2) = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10
    W2[3, 1] = -1 # C11
    
    # Term 2 (M3) = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01
    W2[3, 2] = 1 # C11
    
    # Term 3 (M4) = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C00
    W2[2, 3] = 1 # C10
    
    # Term 4 (M5) = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1 # C01
    
    # Term 5 (M6) = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # Term 6 (M7) = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # Verify Strassen correctness
    # C00: M1(1) + M4(1) - M5(-1) + M7(1) -> 1+1-(-1)+1 = 4? No.
    # C00 = M1 + M4 - M5 + M7.
    # My W2 for M5 is -1. So contribution is -1 * (-1) = 1?
    # Wait, W2[row, col] is the coefficient.
    # If W2[0, 4] = -1, then term 4 contributes -1 * M5.
    # Formula: C00 = 1*M1 + 1*M4 - 1*M5 + 1*M7.
    # So W2[0, 0]=1, W2[0, 3]=1, W2[0, 4]=-1, W2[0, 6]=1. Correct.
    
    # C01: M3 + M5.
    # W2[1, 2]=1, W2[1, 4]=1. Correct.
    
    # C10: M2 + M4.
    # W2[2, 1]=1, W2[2, 3]=1. Correct.
    
    # C11: M1 - M2 + M3 + M6.
    # W2[3, 0]=1, W2[3, 1]=-1, W2[3, 2]=1, W2[3, 5]=1. Correct.
    
    # Construct 6x6 factors
    U6 = np.kron(U3, U2)
    V6 = np.kron(V3, V2)
    W6 = np.kron(W3, W2)
    
    # Convert to lists
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
