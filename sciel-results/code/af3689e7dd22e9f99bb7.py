import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition incrementally.
    # Start with schoolbook, then replace specific blocks with Strassen.
    # However, it's easier to build from scratch.
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Helper to create zero vectors
    def zero_U(): return [0.0] * (n * m)
    def zero_V(): return [0.0] * (m * p)
    def zero_W(): return [0.0] * (p * n)
    
    # Indices for Block 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Strassen terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Note: A indices are (row, col). B indices are (row, col).
    # A00 is A[0,0], A01 is A[0,1], A10 is A[1,0], A11 is A[1,1].
    
    # M1
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+0] = 1; u[1*4+1] = 1
    v[0*5+0] = 1; v[1*5+1] = 1
    w[0*3+0] = 1 # C00
    add_term(u, v, w)
    
    # M2
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+0] = 1; u[1*4+1] = 1
    v[0*5+0] = 1
    w[1*3+0] = 1 # C10
    add_term(u, v, w)
    
    # M3
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+0] = 1
    v[0*5+1] = 1; v[1*5+1] = -1
    w[0*3+1] = 1 # C01
    add_term(u, v, w)
    
    # M4
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+1] = 1
    v[1*5+0] = 1; v[0*5+0] = -1
    w[0*3+0] = 1 # C00
    w[1*3+0] = 1 # C10
    add_term(u, v, w)
    
    # M5
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+0] = 1; u[0*4+1] = 1
    v[1*5+1] = 1
    w[0*3+0] = -1 # C00
    w[0*3+1] = 1  # C01
    add_term(u, v, w)
    
    # M6
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+0] = 1; u[0*4+0] = -1
    v[0*5+0] = 1; v[0*5+1] = 1
    w[1*3+1] = 1 # C11
    add_term(u, v, w)
    
    # M7
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+1] = 1; u[1*4+1] = -1
    v[1*5+0] = 1; v[1*5+1] = 1
    w[0*3+0] = 1  # C00
    w[1*3+1] = 1  # C11
    add_term(u, v, w)
    
    # Block 2: A[0:2, 2:4], B[0:2, 2:4], C[0:2, 2:4]
    # A00 is A[0,2], A01 is A[0,3], A10 is A[1,2], A11 is A[1,3]
    # B00 is B[0,2], B01 is B[0,3], B10 is B[1,2], B11 is B[1,3]
    # C00 is C[0,2], C01 is C[0,3], C10 is C[1,2], C11 is C[1,3]
    
    # M1
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+2] = 1; u[1*4+3] = 1
    v[0*5+2] = 1; v[1*5+3] = 1
    w[2*3+0] = 1 # C02
    add_term(u, v, w)
    
    # M2
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+2] = 1; u[1*4+3] = 1
    v[0*5+2] = 1
    w[2*3+1] = 1 # C12
    add_term(u, v, w)
    
    # M3
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+2] = 1
    v[0*5+3] = 1; v[1*5+3] = -1
    w[3*3+0] = 1 # C03
    add_term(u, v, w)
    
    # M4
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+3] = 1
    v[1*5+2] = 1; v[0*5+2] = -1
    w[2*3+0] = 1 # C02
    w[2*3+1] = 1 # C12
    add_term(u, v, w)
    
    # M5
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+2] = 1; u[0*4+3] = 1
    v[1*5+3] = 1
    w[2*3+0] = -1 # C02
    w[3*3+0] = 1  # C03
    add_term(u, v, w)
    
    # M6
    u = zero_U(); v = zero_V(); w = zero_W()
    u[1*4+2] = 1; u[0*4+2] = -1
    v[0*5+2] = 1; v[0*5+3] = 1
    w[3*3+1] = 1 # C13
    add_term(u, v, w)
    
    # M7
    u = zero_U(); v = zero_V(); w = zero_W()
    u[0*4+3] = 1; u[1*4+3] = -1
    v[1*5+2] = 1; v[1*5+3] = 1
    w[2*3+0] = 1  # C02
    w[3*3+1] = 1  # C13
    add_term(u, v, w)
    
    # Schoolbook for remaining terms
    # Covered: A[0:2, 0:2] with B[0:2, 0:2]
    # Covered: A[0:2, 2:4] with B[0:2, 2:4]
    # Remaining:
    # A[2, 0:4] with B[0:4, 0:5] (Row 2 of A)
    # A[0:2, 0:4] with B[2:4, 0:5] (Rows 2,3 of B)
    # A[0:2, 0:2] with B[0:2, 4:5] (Col 4 of B) - Wait, B cols 0-3 covered?
    # B cols 0-1 covered in Block 1. B cols 2-3 covered in Block 2.
    # So B col 4 is remaining for A[0:2, 0:2] and A[0:2, 2:4].
    # Also A[2, 0:4] with B[0:4, 0:5].
    
    # Let's just iterate all (i,j,k) and add if not covered.
    # Covered set:
    # Block 1: i in 0,1; j in 0,1; k in 0,1
    # Block 2: i in 0,1; j in 2,3; k in 2,3
    
    covered = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                covered.add((i,j,k))
    for i in range(2):
        for j in range(2,4):
            for k in range(2,4):
                covered.add((i,j,k))
                
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    u = zero_U(); v = zero_V(); w = zero_W()
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    add_term(u, v, w)
                    
    rank = len(U)
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    return {"U": U_mat, "V": V_mat, "W": W_mat}
