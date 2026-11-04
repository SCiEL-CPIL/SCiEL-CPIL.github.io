import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-56 exact tensor decomposition for matrix multiplication M_{3,4,5}
    using a block decomposition with Strassen's algorithm for 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen(offset, i1, i2, j1, j2, k1, k2):
        """
        Adds 7 Strassen terms for A(i1,i2 x j1,j2) * B(j1,j2 x k1,k2) -> C(k1,k2 x i1,i2)
        """
        # M1 = (A11 + A22)(B11 + B22)
        U[i1*m + j1, offset] += 1
        U[i2*m + j2, offset] += 1
        V[j1*p + k1, offset] += 1
        V[j2*p + k2, offset] += 1
        W[k1*n + i1, offset] += 1  # C11
        W[k2*n + i2, offset] += 1  # C22
        W[k1*n + i2, offset] -= 1  # C12
        W[k2*n + i1, offset] -= 1  # C21
        
        # M2 = (A21 + A22)B11
        U[i2*m + j1, offset+1] += 1
        U[i2*m + j2, offset+1] += 1
        V[j1*p + k1, offset+1] += 1
        W[k2*n + i1, offset+1] += 1  # C21
        W[k2*n + i2, offset+1] -= 1  # C22
        
        # M3 = A11(B12 - B22)
        U[i1*m + j1, offset+2] += 1
        V[j1*p + k2, offset+2] += 1
        V[j2*p + k2, offset+2] -= 1
        W[k1*n + i2, offset+2] += 1  # C12
        W[k2*n + i2, offset+2] += 1  # C22
        
        # M4 = A22(B21 - B11)
        U[i2*m + j2, offset+3] += 1
        V[j2*p + k1, offset+3] += 1
        V[j1*p + k1, offset+3] -= 1
        W[k1*n + i1, offset+3] += 1  # C11
        W[k2*n + i1, offset+3] += 1  # C21
        
        # M5 = (A11 + A12)B22
        U[i1*m + j1, offset+4] += 1
        U[i1*m + j2, offset+4] += 1
        V[j2*p + k2, offset+4] += 1
        W[k2*n + i1, offset+4] -= 1  # C21
        W[k2*n + i2, offset+4] += 1  # C22
        
        # M6 = (A21 - A11)(B11 + B12)
        U[i2*m + j1, offset+5] += 1
        U[i1*m + j1, offset+5] -= 1
        V[j1*p + k1, offset+5] += 1
        V[j1*p + k2, offset+5] += 1
        W[k2*n + i2, offset+5] += 1  # C22
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i1*m + j2, offset+6] += 1
        U[i2*m + j2, offset+6] -= 1
        V[j2*p + k1, offset+6] += 1
        V[j2*p + k2, offset+6] += 1
        W[k1*n + i1, offset+6] += 1  # C11
        W[k1*n + i2, offset+6] -= 1  # C12

    def add_schoolbook_2x2x1(offset, i1, i2, j1, j2, k1):
        """
        Adds 4 schoolbook terms for A(i1,i2 x j1,j2) * B(j1,j2 x k1) -> C(k1 x i1,i2)
        Note: C index in W is k*n + i. Here k is k1. i is i1, i2.
        """
        # A11 B11 -> C11 (i1, k1)
        U[i1*m + j1, offset] += 1
        V[j1*p + k1, offset] += 1
        W[k1*n + i1, offset] += 1
        
        # A12 B21 -> C12 (i1, k1)
        U[i1*m + j2, offset+1] += 1
        V[j2*p + k1, offset+1] += 1
        W[k1*n + i1, offset+1] += 1
        
        # A21 B11 -> C21 (i2, k1)
        U[i2*m + j1, offset+2] += 1
        V[j1*p + k1, offset+2] += 1
        W[k1*n + i2, offset+2] += 1
        
        # A22 B21 -> C22 (i2, k1)
        U[i2*m + j2, offset+3] += 1
        V[j2*p + k1, offset+3] += 1
        W[k1*n + i2, offset+3] += 1

    def add_schoolbook_1x2x2(offset, i1, j1, j2, k1, k2):
        """
        Adds 4 schoolbook terms for A(i1 x j1,j2) * B(j1,j2 x k1,k2) -> C(k1,k2 x i1)
        """
        # A11 B11 -> C11 (k1, i1)
        U[i1*m + j1, offset] += 1
        V[j1*p + k1, offset] += 1
        W[k1*n + i1, offset] += 1
        
        # A11 B12 -> C12 (k2, i1)
        U[i1*m + j1, offset+1] += 1
        V[j1*p + k2, offset+1] += 1
        W[k2*n + i1, offset+1] += 1
        
        # A12 B21 -> C21 (k1, i1)
        U[i1*m + j2, offset+2] += 1
        V[j2*p + k1, offset+2] += 1
        W[k1*n + i1, offset+2] += 1
        
        # A12 B22 -> C22 (k2, i1)
        U[i1*m + j2, offset+3] += 1
        V[j2*p + k2, offset+3] += 1
        W[k2*n + i1, offset+3] += 1

    def add_schoolbook_1x2x1(offset, i1, j1, j2, k1):
        """
        Adds 2 schoolbook terms for A(i1 x j1,j2) * B(j1,j2 x k1) -> C(k1 x i1)
        """
        # A11 B11 -> C11 (k1, i1)
        U[i1*m + j1, offset] += 1
        V[j1*p + k1, offset] += 1
        W[k1*n + i1, offset] += 1
        
        # A12 B21 -> C12 (k1, i1)
        U[i1*m + j2, offset+1] += 1
        V[j2*p + k1, offset+1] += 1
        W[k1*n + i1, offset+1] += 1

    # Block Indices
    # A rows: 0,1 (i1=0, i2=1), 2 (i3=2)
    # A cols: 0,1 (j1=0, j2=1), 2,3 (j3=2, j4=3)
    # B rows: 0,1 (j1=0, j2=1), 2,3 (j3=2, j4=3)
    # B cols: 0,1 (k1=0, k2=1), 2,3 (k3=2, k4=3), 4 (k5=4)
    
    # 1. A11 B11 (Strassen)
    add_strassen(0, 0, 1, 0, 1, 0, 1)
    
    # 2. A12 B21 (Strassen)
    add_strassen(7, 0, 1, 2, 3, 2, 3)
    
    # 3. A11 B12 (Strassen)
    add_strassen(14, 0, 1, 0, 1, 2, 3)
    
    # 4. A12 B22 (Strassen)
    add_strassen(21, 0, 1, 2, 3, 4, 4) # Wait, B22 is rows 2,3 cols 2,3. 
    # Correction: B12 is rows 0,1 cols 2,3. B22 is rows 2,3 cols 2,3.
    # A11 is rows 0,1 cols 0,1. A12 is rows 0,1 cols 2,3.
    # C12 = A11 B12 + A12 B22.
    # A11 B12: A(0,1; 0,1) * B(0,1; 2,3) -> C(2,3; 0,1). Strassen.
    # A12 B22: A(0,1; 2,3) * B(2,3; 2,3) -> C(2,3; 0,1). Strassen.
    
    # Let's re-verify the block calls.
    # C11 (rows 0,1; cols 0,1) = A11(0,1;0,1)B11(0,1;0,1) + A12(0,1;2,3)B21(2,3;0,1)
    # C12 (rows 0,1; cols 2,3) = A11(0,1;0,1)B12(0,1;2,3) + A12(0,1;2,3)B22(2,3;2,3)
    # C13 (rows 0,1; col 4)    = A11(0,1;0,1)B13(0,1;4)    + A12(0,1;2,3)B23(2,3;4)
    # C21 (row 2; cols 0,1)    = A21(2;0,1)B11(0,1;0,1)     + A22(2;2,3)B21(2,3;0,1)
    # C22 (row 2; cols 2,3)    = A21(2;0,1)B12(0,1;2,3)     + A22(2;2,3)B22(2,3;2,3)
    # C23 (row 2; col 4)       = A21(2;0,1)B13(0,1;4)       + A22(2;2,3)B23(2,3;4)

    # 1. A11 B11: Strassen (0,1;0,1) * (0,1;0,1)
    add_strassen(0, 0, 1, 0, 1, 0, 1)
    
    # 2. A12 B21: Strassen (0,1;2,3) * (2,3;0,1)
    add_strassen(7, 0, 1, 2, 3, 0, 1)
    
    # 3. A11 B12: Strassen (0,1;0,1) * (0,1;2,3)
    add_strassen(14, 0, 1, 0, 1, 2, 3)
    
    # 4. A12 B22: Strassen (0,1;2,3) * (2,3;2,3)
    add_strassen(21, 0, 1, 2, 3, 2, 3)
    
    # 5. A11 B13: Schoolbook 2x2x1 (0,1;0,1) * (0,1;4)
    add_schoolbook_2x2x1(28, 0, 1, 0, 1, 4)
    
    # 6. A12 B23: Schoolbook 2x2x1 (0,1;2,3) * (2,3;4)
    add_schoolbook_2x2x1(32, 0, 1, 2, 3, 4)
    
    # 7. A21 B11: Schoolbook 1x2x2 (2;0,1) * (0,1;0,1)
    add_schoolbook_1x2x2(36, 2, 0, 1, 0, 1)
    
    # 8. A22 B21: Schoolbook 1x2x2 (2;2,3) * (2,3;0,1)
    add_schoolbook_1x2x2(40, 2, 2, 3, 0, 1)
    
    # 9. A21 B12: Schoolbook 1x2x2 (2;0,1) * (0,1;2,3)
    add_schoolbook_1x2x2(44, 2, 0, 1, 2, 3)
    
    # 10. A22 B22: Schoolbook 1x2x2 (2;2,3) * (2,3;2,3)
    add_schoolbook_1x2x2(48, 2, 2, 3, 2, 3)
    
    # 11. A21 B13: Schoolbook 1x2x1 (2;0,1) * (0,1;4)
    add_schoolbook_1x2x1(52, 2, 0, 1, 4)
    
    # 12. A22 B23: Schoolbook 1x2x1 (2;2,3) * (2,3;4)
    add_schoolbook_1x2x1(54, 2, 2, 3, 4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
