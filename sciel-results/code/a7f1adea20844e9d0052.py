import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 bilinear forms for 2x2 matrix multiplication
    # A is 2x2, B is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A01-A10)(B00+B01)
    # M7 = (A10-A00)(B11+B01)
    
    # We have two blocks of B: B1 (cols 0,1) and B2 (cols 2,3)
    # Block 1: r = 0..6
    # Block 2: r = 7..13
    
    def set_strassen_block(offset, col_offset):
        # col_offset is 0 for B1, 2 for B2
        # Indices for A: i*m + j
        # Indices for B: j*p + (col_offset + k)
        # Indices for C: (col_offset + k)*n + i
        
        # M1
        r = offset
        U[0*m + 0, r] += 1; U[1*m + 1, r] += 1
        V[0*p + col_offset, r] += 1; V[1*p + col_offset + 1, r] += 1
        W[col_offset*n + 0, r] += 1; W[(col_offset+1)*n + 0, r] += 1
        W[col_offset*n + 1, r] -= 1; W[(col_offset+1)*n + 1, r] -= 1
        # Wait, W corresponds to C_{k,i}.
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5
        
        # Let's define the W coefficients for each M
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1
        # M3: C01 += 1, C11 -= 1
        # M4: C00 -= 1, C10 += 1
        # M5: C01 += 1, C11 += 1
        # M6: (Unused in standard Strassen output formulas above? 
        #      Standard Strassen:
        #      C00 = M1 - M4 + M7
        #      C01 = M3 + M5 + M7
        #      C10 = M2 + M4
        #      C11 = M1 - M3 + M5
        #      M6 is not used in the final sum?
        #      Ah, M6 is (A01-A10)(B00+B01).
        #      It is not in the standard 7-product Strassen output formulas listed above?
        #      Let's check.
        #      Usually:
        #      M1=(A11+A22)(B11+B22)
        #      M2=(A21+A22)B11
        #      M3=A11(B21-B22)
        #      M4=A22(B12-B11)
        #      M5=(A11+A12)B22
        #      M6=(A21-A11)(B11+B12)
        #      M7=(A12-A22)(B21+B22)
        #      C11=M1-M4+M7
        #      C12=M3+M5+M7
        #      C21=M2+M4
        #      C22=M1-M3+M5
        #      Yes, M6 is not used in the final reconstruction?
        #      Wait, that's 7 products.
        #      If M6 is not used, why compute it?
        #      Ah, the standard Strassen uses 7 products.
        #      Let's re-verify the formulas.
        #      Wikipedia:
        #      p1 = (a11 + a22)(b11 + b22)
        #      p2 = (a21 + a22)b11
        #      p3 = a11(b21 - b22)
        #      p4 = a22(b12 - b11)
        #      p5 = (a11 + a12)b22
        #      p6 = (a21 - a11)(b11 + b12)
        #      p7 = (a12 - a22)(b21 + b22)
        #      c11 = p1 - p4 + p7
        #      c12 = p3 + p5 + p7
        #      c21 = p2 + p4
        #      c22 = p1 - p3 + p5
        #      Yes, p6 is not in the final sum.
        #      This is correct.
        
        # So for each block, we define 7 terms.
        
        # Term 0 (M1)
        r = offset + 0
        U[0, r] = 1; U[3, r] = 1  # A00, A11
        V[col_offset, r] = 1; V[col_offset + 1 + p, r] = 1 # B00, B11 (j=0,k=0; j=1,k=1)
        # W indices: k*n + i
        # C00 (k=0, i=0): idx 0
        # C11 (k=1, i=1): idx 1*n+1 = 3 (relative to block? No, global)
        # Global W index for C_{i,k} is k*n + i.
        # For block 1 (cols 0,1):
        # C_{0,0} -> 0*2+0=0
        # C_{1,0} -> 0*2+1=1
        # C_{0,1} -> 1*2+0=2
        # C_{1,1} -> 1*2+1=3
        W[col_offset*n + 0, r] = 1 # C00
        W[(col_offset+1)*n + 1, r] = 1 # C11
        
        # Term 1 (M2)
        r = offset + 1
        U[2, r] = 1; U[3, r] = 1 # A10, A11
        V[col_offset, r] = 1 # B00
        W[col_offset*n + 1, r] = 1 # C10
        
        # Term 2 (M3)
        r = offset + 2
        U[0, r] = 1 # A00
        V[col_offset + 1, r] = 1; V[col_offset + 1 + p, r] = -1 # B01, -B11
        W[(col_offset+1)*n + 0, r] = 1 # C01
        W[(col_offset+1)*n + 1, r] = -1 # C11
        
        # Term 3 (M4)
        r = offset + 3
        U[3, r] = 1 # A11
        V[col_offset + 1 + p, r] = 1; V[col_offset, r] = -1 # B10, -B00
        W[col_offset*n + 0, r] = -1 # C00
        W[col_offset*n + 1, r] = 1 # C10
        
        # Term 4 (M5)
        r = offset + 4
        U[0, r] = 1; U[1, r] = 1 # A00, A01
        V[col_offset + 1 + p, r] = 1 # B11
        W[(col_offset+1)*n + 0, r] = 1 # C01
        W[(col_offset+1)*n + 1, r] = 1 # C11
        
        # Term 5 (M6)
        r = offset + 5
        U[1, r] = 1; U[2, r] = -1 # A01, -A10
        V[col_offset, r] = 1; V[col_offset + 1, r] = 1 # B00, B01
        # W is 0 for M6
        
        # Term 6 (M7)
        r = offset + 6
        U[2, r] = 1; U[0, r] = -1 # A10, -A00
        V[col_offset + 1 + p, r] = 1; V[col_offset + 1, r] = 1 # B11, B01
        W[col_offset*n + 0, r] = 1 # C00
        W[(col_offset+1)*n + 0, r] = 1 # C01

    set_strassen_block(0, 0)
    set_strassen_block(7, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
