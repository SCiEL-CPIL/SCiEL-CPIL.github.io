import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen factors
    # U2: 4x7, indices i*2+j
    # V2: 4x7, indices j*2+k
    # W2: 4x7, indices k*2+i
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11)(B00 + B11)
    # A00: 0, A11: 3 | B00: 0, B11: 3 | C00: 0, C11: 3
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # A10: 2, A11: 3 | B00: 0 | C10: 1 (+), C11: 3 (-)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    # A00: 0 | B01: 1, B11: 3 | C01: 2 (+), C11: 3 (+)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    # A11: 3 | B10: 2, B00: 0 | C00: 0 (+), C10: 1 (+)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A00 + A01)B11
    # A00: 0, A01: 1 | B11: 3 | C00: 0 (-), C01: 2 (+)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10: 2, A00: 0 | B00: 0, B01: 1 | C11: 3 (+)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01: 1, A11: 3 | B10: 2, B11: 3 | C00: 0 (+)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Compute Kronecker products for 4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The problem requires integer/half-integer entries. 
    # The Kronecker product of integers is integers.
    # We return lists of lists as per the interface example, or numpy arrays?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returned lists of lists. I will return lists of lists to be safe, 
    # but numpy arrays are often accepted if they can be converted. 
    # Let's stick to the seed's format: lists of lists.
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
# EVOLVE-BLOCK-END
