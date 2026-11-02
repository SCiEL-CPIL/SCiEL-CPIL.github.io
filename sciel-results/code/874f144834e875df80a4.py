import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication (Strassen).
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 intermediate products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices:
    # U rows (A): 0=A00, 1=A01, 2=A10, 3=A11
    # V rows (B): 0=B00, 1=B01, 2=B10, 3=B11
    # W rows (C, trace convention k*n+i):
    #   C00 (i=0,k=0) -> 0
    #   C10 (i=1,k=0) -> 1
    #   C01 (i=0,k=1) -> 2
    #   C11 (i=1,k=1) -> 3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Col 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Col 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    # Convert to list of lists of integers/half-integers
    def convert_to_int_if_close(mat):
        return [
            [int(round(v)) if abs(v - round(v)) < 1e-9 else float(v) for v in row]
            for row in mat
        ]

    return {
        "U": convert_to_int_if_close(U),
        "V": convert_to_int_if_close(V),
        "W": convert_to_int_if_close(W)
    }
# EVOLVE-BLOCK-END
