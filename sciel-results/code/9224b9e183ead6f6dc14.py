import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a trivial 1x1x1 or similar simple block?
    # Actually, let's just hardcode the loops for the specific blocks identified.
    
    # Block 1: A[0:2, 2] * B[2:3, 0:7] -> C[0:2, 0:7]
    # A col 2, B row 2.
    # i in 0..1, k in 0..6
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2, col] = 1.0
            V[2 * 7 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
            col += 1
            
    # Block 2: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    # Strassen indices:
    # M1: (A11+A22)(B11+B22)
    # A11: U[0*3+0], A22: U[1*3+1]
    # B11: V[0*7+0], B22: V[1*7+1]
    # C11: W[0*2+0], C12: W[0*2+1], C21: W[1*2+0], C22: W[1*2+1]
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # a_r, a_c: top-left row/col of A submatrix (2x2)
        # b_r, b_c: top-left row/col of B submatrix (2x2)
        # c_r, c_c: top-left row/col of C submatrix (2x2)
        
        # A indices
        A11 = (a_r + 0) * 3 + (a_c + 0)
        A12 = (a_r + 0) * 3 + (a_c + 1)
        A21 = (a_r + 1) * 3 + (a_c + 0)
        A22 = (a_r + 1) * 3 + (a_c + 1)
        
        # B indices
        B11 = (b_r + 0) * 7 + (b_c + 0)
        B12 = (b_r + 0) * 7 + (b_c + 1)
        B21 = (b_r + 1) * 7 + (b_c + 0)
        B22 = (b_r + 1) * 7 + (b_c + 1)
        
        # C indices
        C11 = (c_r + 0) * 2 + (c_c + 0)
        C12 = (c_r + 0) * 2 + (c_c + 1)
        C21 = (c_r + 1) * 2 + (c_c + 0)
        C22 = (c_r + 1) * 2 + (c_c + 1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[A11, start_col] += 1.0
        U[A22, start_col] += 1.0
        V[B11, start_col] += 1.0
        V[B22, start_col] += 1.0
        W[C11, start_col] += 1.0
        W[C22, start_col] += 1.0
        
        # M2 = (A21 + A22) * B11
        U[A21, start_col + 1] += 1.0
        U[A22, start_col + 1] += 1.0
        V[B11, start_col + 1] += 1.0
        W[C21, start_col + 1] += 1.0
        W[C22, start_col + 1] += -1.0
        
        # M3 = A11 * (B12 - B22)
        U[A11, start_col + 2] += 1.0
        V[B12, start_col + 2] += 1.0
        V[B22, start_col + 2] += -1.0
        W[C12, start_col + 2] += 1.0
        W[C22, start_col + 2] += 1.0
        
        # M4 = A22 * (B21 - B11)
        U[A22, start_col + 3] += 1.0
        V[B21, start_col + 3] += 1.0
        V[B11, start_col + 3] += -1.0
        W[C11, start_col + 3] += 1.0
        W[C21, start_col + 3] += 1.0
        
        # M5 = (A11 + A12) * B22
        U[A11, start_col + 4] += 1.0
        U[A12, start_col + 4] += 1.0
        V[B22, start_col + 4] += 1.0
        W[C11, start_col + 4] += -1.0
        W[C12, start_col + 4] += 1.0
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[A21, start_col + 5] += 1.0
        U[A11, start_col + 5] += -1.0
        V[B11, start_col + 5] += 1.0
        V[B12, start_col + 5] += 1.0
        W[C22, start_col + 5] += 1.0
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[A12, start_col + 6] += 1.0
        U[A22, start_col + 6] += -1.0
        V[B21, start_col + 6] += 1.0
        V[B22, start_col + 6] += 1.0
        W[C11, start_col + 6] += 1.0
        W[C21, start_col + 6] += -1.0
        
        return start_col + 7

    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 0, 0, 4, 0, 4, col)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 6:7] -> C[0:2, 6:7]
    # Trivial 2x2x1
    for i in range(2):
        for j in range(2):
            U[i * 3 + j, col] = 1.0
            V[j * 7 + 6, col] = 1.0
            W[6 * 2 + i, col] = 1.0
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
