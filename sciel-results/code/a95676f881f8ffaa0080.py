import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    # Returns U, V, W for 2x2 Strassen, rank 7
    # U, V, W shapes (4, 7)
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = 1 # C00=M1+M4-M5+M7, C11=M1-M2+M3+M6
    # Wait, W must be the coefficient for the output C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's fill W carefully.
    # Row 0 of W is C00. Row 1 is C01. Row 2 is C10. Row 3 is C11.
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    # Returns U, V, W for 3x3 Schoolbook, rank 27
    U = np.zeros((9, 27), dtype=int)
    V = np.zeros((9, 27), dtype=int)
    W = np.zeros((9, 27), dtype=int)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    U3, V3, W3 = _schoolbook_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker products
    U6 = np.kron(U3, U2)
    V6 = np.kron(V3, V2)
    W6 = np.kron(W3, W2)
    
    # Shapes are (36, 189)
    
    # Permute rows to match global indexing
    # U6 row r corresponds to (i3, j3, i2, j2) where r = (i3*3+j3)*4 + (i2*2+j2)
    # Global Index I = 2*i3 + i2, J = 2*j3 + j2
    # Target Row T = I*6 + J
    
    def permute_rows(M, n_outer, n_inner, outer_mult, inner_mult):
        # n_outer: 3, n_inner: 2
        # M shape: (n_outer*n_inner*n_outer*n_inner, rank) ? No.
        # M shape: (9*4, rank) = (36, rank)
        # r = r_outer * 4 + r_inner
        # r_outer = i_o * 3 + j_o
        # r_inner = i_i * 2 + j_i
        # Global I = outer_mult * i_o + inner_mult * i_i ?
        # For U: I = 2*i3 + i2, J = 2*j3 + j2
        # T = I*6 + J = (2*i3 + i2)*6 + (2*j3 + j2) = 12*i3 + 6*i2 + 2*j3 + j2
        
        rows, cols = M.shape
        P = np.zeros((rows, rows), dtype=int)
        for r in range(rows):
            r_outer = r // 4
            r_inner = r % 4
            i3 = r_outer // 3
            j3 = r_outer % 3
            i2 = r_inner // 2
            j2 = r_inner % 2
            
            I = 2 * i3 + i2
            J = 2 * j3 + j2
            T = I * 6 + J
            
            P[T, r] = 1
        
        return P @ M

    U_final = permute_rows(U6, 3, 2, 2, 1)
    V_final = permute_rows(V6, 3, 2, 2, 1)
    W_final = permute_rows(W6, 3, 2, 2, 1)
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
