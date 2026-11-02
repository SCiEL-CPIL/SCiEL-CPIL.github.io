import numpy as np

def construct():
    n, m, p = 2, 4, 5
    
    # Calculate exact rank needed to avoid waste or overflow
    # 4 Strassen blocks (7 each) + 2 Schoolbook 2x1 blocks (4 each) = 28 + 8 = 36
    # Add a small buffer just in case, but keep it tight.
    max_rank = 40
    U = np.zeros((n*m, max_rank), dtype=float)
    V = np.zeros((m*p, max_rank), dtype=float)
    W = np.zeros((p*n, max_rank), dtype=float)
    
    col = 0
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # Indices in U (flattened A)
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        # Indices in V (flattened B)
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        # Indices in W (flattened C, trace convention W[k*n+i, r])
        # C is 2x5. C[r, c]. Index in W is c*n + r.
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00, col] += 1
        U[u11, col] += 1
        V[v00, col] += 1
        V[v11, col] += 1
        W[w00, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10, col] += 1
        U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1
        W[w11, col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00, col] += 1
        V[v01, col] += 1
        V[v11, col] -= 1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11, col] += 1
        V[v10, col] += 1
        V[v00, col] -= 1
        W[w00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00, col] += 1
        U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1
        W[w01, col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10, col] += 1
        U[u00, col] -= 1
        V[v00, col] += 1
        V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01, col] += 1
        U[u11, col] -= 1
        V[v10, col] += 1
        V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v0 = b_r0 * p + b_c0
        v1 = (b_r0 + 1) * p + b_c0
        
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        U[u00, col] = 1
        V[v0, col] = 1
        W[w0, col] = 1
        col += 1
        
        U[u01, col] = 1
        V[v1, col] = 1
        W[w0, col] = 1
        col += 1
        
        U[u10, col] = 1
        V[v0, col] = 1
        W[w1, col] = 1
        col += 1
        
        U[u11, col] = 1
        V[v1, col] = 1
        W[w1, col] = 1
        col += 1

    # Block 1: Columns 0,1 of C. A (2x4) * B[:,0:2] (4x2)
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block 2: Columns 2,3 of C. A (2x4) * B[:,2:4] (4x2)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block 3: Column 4 of C. A (2x4) * B[:,4] (4x1)
    add_schoolbook_2x1(0, 0, 0, 4, 0, 4)
    add_schoolbook_2x1(0, 2, 2, 4, 0, 4)
    
    # Trim to actual used rank
    actual_rank = col
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
    
    return {"U": U, "V": V, "W": W}
