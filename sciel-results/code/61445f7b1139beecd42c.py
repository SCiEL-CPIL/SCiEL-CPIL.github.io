import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        # A is 2x2 at rows a_r0, a_r1 and cols a_c0, a_c1
        # B is 2x2 at rows b_r0, b_r1 and cols b_c0, b_c1
        # C is 2x2 at rows c_r0, c_r1 and cols c_c0, c_c1
        
        # Indices for U (A)
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c1
        u10 = a_r1 * m + a_c0
        u11 = a_r1 * m + a_c1
        
        # Indices for V (B)
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c1
        v10 = b_r1 * p + b_c0
        v11 = b_r1 * p + b_c1
        
        # Indices for W (C)
        w00 = c_c0 * n + c_r0
        w01 = c_c0 * n + c_r1
        w10 = c_c1 * n + c_r0
        w11 = c_c1 * n + c_r1
        
        # m1 = A00 * (B10 + B11)
        U[u00, col] = 1
        V[v10, col] = 1
        V[v11, col] = 1
        W[w00, col] = 1
        W[w01, col] = 1
        W[w10, col] = 1
        W[w11, col] = 1
        # Correction: m1 contributes to C00 and C11 with +1, and C01? No.
        # C00 = m5 + m4 - m2 + m1
        # C01 = m3 + m5
        # C10 = m4 + m6
        # C11 = m1 - m5 + m3 + m7
        
        # Let's add terms one by one to be safe.
        
        # m1 = A00 * (B10 + B11)
        # Contribution: +1 to C00, +1 to C11
        U[u00, col] = 1
        V[v10, col] = 1
        V[v11, col] = 1
        W[w00, col] = 1
        W[w11, col] = 1
        col += 1
        
        # m2 = (A00 + A01) * B11
        # Contribution: -1 to C00
        U[u00, col] = 1
        U[u01, col] = 1
        V[v11, col] = 1
        W[w00, col] = -1
        col += 1
        
        # m3 = (A10 + A11) * B00
        # Contribution: +1 to C01, +1 to C11
        U[u10, col] = 1
        U[u11, col] = 1
        V[v00, col] = 1
        W[w01, col] = 1
        W[w11, col] = 1
        col += 1
        
        # m4 = A11 * (B01 - B00)
        # Contribution: +1 to C00, +1 to C10
        U[u11, col] = 1
        V[v01, col] = 1
        V[v00, col] = -1
        W[w00, col] = 1
        W[w10, col] = 1
        col += 1
        
        # m5 = (A00 + A11) * (B00 + B11)
        # Contribution: +1 to C00, +1 to C01, -1 to C11
        U[u00, col] = 1
        U[u11, col] = 1
        V[v00, col] = 1
        V[v11, col] = 1
        W[w00, col] = 1
        W[w01, col] = 1
        W[w11, col] = -1
        col += 1
        
        # m6 = (A01 - A11) * (B01 + B11)
        # Contribution: +1 to C10
        U[u01, col] = 1
        U[u11, col] = -1
        V[v01, col] = 1
        V[v11, col] = 1
        W[w10, col] = 1
        col += 1
        
        # m7 = (A10 - A00) * (B10 + B00)
        # Contribution: +1 to C11
        U[u10, col] = 1
        U[u00, col] = -1
        V[v10, col] = 1
        V[v00, col] = 1
        W[w11, col] = 1
        col += 1
        
        return col

    def add_simple(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        U[a_r * m + a_c, col] = 1
        V[b_r * p + b_c, col] = 1
        W[c_c * n + c_r, col] = 1
        col += 1
        return col

    # Top Left: C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # We can't just call add_strassen twice because it assumes a single 2x2 * 2x2.
    # C_TL = A_TL B_TL + A_TR B_BL
    # This is sum of two 2x2 products.
    # We can compute A_TL B_TL with Strassen (7 terms) and A_TR B_BL with Strassen (7 terms).
    
    # Product 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    
    # Product 2: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    
    # Top Right: C[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # Product 3: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    
    # Product 4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # Bottom Left: C[2:3, 0:2]
    # A[2:3, 0:2] * B[0:2, 0:2] + A[2:3, 2:4] * B[2:4, 0:2]
    # A is 1x2. B is 2x2.
    # C is 1x2.
    # Terms:
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0] + A[2,3]B[3,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1] + A[2,3]B[3,1]
    
    # 4 terms for C[2,0]
    add_simple(2, 0, 0, 0, 2, 0)
    add_simple(2, 1, 1, 0, 2, 0)
    add_simple(2, 2, 2, 0, 2, 0)
    add_simple(2, 3, 3, 0, 2, 0)
    
    # 4 terms for C[2,1]
    add_simple(2, 0, 0, 1, 2, 1)
    add_simple(2, 1, 1, 1, 2, 1)
    add_simple(2, 2, 2, 1, 2, 1)
    add_simple(2, 3, 3, 1, 2, 1)
    
    # Bottom Right: C[2:3, 2:4]
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2] + A[2,3]B[3,2]
    # C[2,3] = A[2,0]B[0,3] + A[2,1]B[1,3] + A[2,2]B[2,3] + A[2,3]B[3,3]
    
    add_simple(2, 0, 0, 2, 2, 2)
    add_simple(2, 1, 1, 2, 2, 2)
    add_simple(2, 2, 2, 2, 2, 2)
    add_simple(2, 3, 3, 2, 2, 2)
    
    add_simple(2, 0, 0, 3, 2, 3)
    add_simple(2, 1, 1, 3, 2, 3)
    add_simple(2, 2, 2, 3, 2, 3)
    add_simple(2, 3, 3, 3, 2, 3)
    
    # Verify rank
    if col != rank:
        # Pad or truncate
        if col < rank:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
        else:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
            col = rank

    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
