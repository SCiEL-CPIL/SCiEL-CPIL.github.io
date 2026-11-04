import numpy as np

DIMENSIONS = [4, 4, 6]

def construct_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing:
    U[i*2+j, k]
    V[j*2+k, l]
    W[k*2+i, l]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 1: (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # c11 += m1
    W[3, 0] = 1  # c22 += m1
    
    # Term 2: (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1 # c22 -= m2
    
    # Term 3: a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # c12 += m3
    W[3, 2] = 1  # c22 += m3
    
    # Term 4: a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # c11 += m4
    W[3, 3] = 1  # c21 += m4
    
    # Term 5: (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # c11 -= m5
    W[1, 4] = 1  # c12 += m5
    
    # Term 6: (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c22 += m6
    
    # Term 7: (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c11 += m7
    
    return U, V, W

def construct_444():
    """
    Returns U, V, W for 4x4x4 matrix multiplication (rank 49).
    Composed from 7 instances of 2x2x2.
    """
    U2, V2, W2 = construct_222()
    
    # Strassen coefficients for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # For each super term k (0-6), define:
    # A_coeffs: dict from (r,c) block to coeff
    # B_coeffs: dict from (r,c) block to coeff
    # C_coeffs: dict from (r,c) block to coeff
    
    A_coeffs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    B_coeffs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    C_coeffs = [
        { (0,0): 1, (1,1): 1 }, # M1 -> C00, C11
        { (1,0): 1 },           # M2 -> C10
        { (0,1): 1, (1,1): 1 }, # M3 -> C01, C11
        { (0,0): 1, (1,0): 1 }, # M4 -> C00, C10
        { (0,0): -1, (0,1): 1 },# M5 -> C00, C01
        { (1,1): 1 },           # M6 -> C11
        { (0,0): 1 }            # M7 -> C00
    ]
    
    # Wait, let's re-verify C_coeffs from Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1 },           # M2
        { (0,1): 1, (1,1): 1 }, # M3
        { (0,0): 1, (1,0): 1 }, # M4
        { (0,0): -1, (0,1): 1 },# M5
        { (1,1): 1 },           # M6
        { (0,0): 1 }            # M7
    ]
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for k in range(7):
        col_base = k * 7
        for l in range(7):
            col = col_base + l
            
            # U contributions
            # U2[i'*2+j', l] is coeff for A'_k[i', j']
            for i_prime in range(2):
                for j_prime in range(2):
                    coeff_inner = U2[i_prime * 2 + j_prime, l]
                    if coeff_inner == 0: continue
                    # A'_k[i', j'] = sum_{r,c} A_coeffs[k][(r,c)] * A[r,c][i', j']
                    for (r, c), coeff_outer in A_coeffs[k].items():
                        row_A = r * 2 + i_prime
                        col_A = c * 2 + j_prime
                        U[row_A * 4 + col_A, col] += coeff_outer * coeff_inner
            
            # V contributions
            for i_prime in range(2): # j in 2x2
                for j_prime in range(2): # k in 2x2
                    coeff_inner = V2[i_prime * 2 + j_prime, l]
                    if coeff_inner == 0: continue
                    for (r, c), coeff_outer in B_coeffs[k].items():
                        row_B = r * 2 + i_prime
                        col_B = c * 2 + j_prime
                        V[row_B * 4 + col_B, col] += coeff_outer * coeff_inner
            
            # W contributions
            for i_prime in range(2): # i in 2x2 (row of C block)
                for j_prime in range(2): # j in 2x2 (col of C block)
                    coeff_inner = W2[j_prime * 2 + i_prime, l]
                    if coeff_inner == 0: continue
                    for (r, c), coeff_outer in C_coeffs[k].items():
                        row_C = r * 2 + i_prime
                        col_C = c * 2 + j_prime
                        # W index: k*n + i. Here n=4.
                        # W[col_C * 4 + row_C, col]
                        W[col_C * 4 + row_C, col] += coeff_outer * coeff_inner
                        
    return U, V, W

def construct_442():
    """
    Returns U, V, W for 4x4x2 matrix multiplication (rank 28).
    Composed from 4 instances of 2x2x2.
    A is 4x4, B is 4x2.
    Split A into 2x2 blocks: A00, A01, A10, A11.
    Split B into 2x2 blocks: B0, B1 (top, bottom).
    C00 = A00 B0 + A01 B1
    C10 = A10 B0 + A11 B1
    """
    U2, V2, W2 = construct_222()
    
    # 4 products:
    # P0 = A00 B0 -> C00
    # P1 = A01 B1 -> C00
    # P2 = A10 B0 -> C10
    # P3 = A11 B1 -> C10
    
    A_coeffs = [
        { (0,0): 1 }, # P0
        { (0,1): 1 }, # P1
        { (1,0): 1 }, # P2
        { (1,1): 1 }  # P3
    ]
    
    # B blocks: B0 is rows 0-1, B1 is rows 2-3.
    # B is 4x2.
    # B0 is 2x2 (rows 0-1, cols 0-1).
    # B1 is 2x2 (rows 2-3, cols 0-1).
    # In 442, B is 4x2.
    # Block (r, c) of B?
    # B has 2 rows of blocks (0, 1) and 1 col of blocks (0).
    # B0 is block (0,0). B1 is block (1,0).
    
    B_coeffs = [
        { (0,0): 1 }, # P0 uses B0
        { (1,0): 1 }, # P1 uses B1
        { (0,0): 1 }, # P2 uses B0
        { (1,0): 1 }  # P3 uses B1
    ]
    
    C_coeffs = [
        { (0,0): 1 }, # P0 -> C00 (block 0,0)
        { (0,0): 1 }, # P1 -> C00
        { (1,0): 1 }, # P2 -> C10 (block 1,0)
        { (1,0): 1 }  # P3 -> C10
    ]
    
    U = np.zeros((16, 28), dtype=int)
    V = np.zeros((8, 28), dtype=int) # B is 4x2 -> 8 entries
    W = np.zeros((8, 28), dtype=int) # C is 4x2 -> 8 entries
    
    for k in range(4):
        col_base = k * 7
        for l in range(7):
            col = col_base + l
            
            # U contributions
            for i_prime in range(2):
                for j_prime in range(2):
                    coeff_inner = U2[i_prime * 2 + j_prime, l]
                    if coeff_inner == 0: continue
                    for (r, c), coeff_outer in A_coeffs[k].items():
                        row_A = r * 2 + i_prime
                        col_A = c * 2 + j_prime
                        U[row_A * 4 + col_A, col] += coeff_outer * coeff_inner
            
            # V contributions
            # B is 4x2.
            # Block (r, c) of B covers rows 2r..2r+1, cols 2c..2c+1.
            # Here B has only 1 col of blocks (c=0).
            for i_prime in range(2): # row in block
                for j_prime in range(2): # col in block
                    coeff_inner = V2[i_prime * 2 + j_prime, l]
                    if coeff_inner == 0: continue
                    for (r, c), coeff_outer in B_coeffs[k].items():
                        row_B = r * 2 + i_prime
                        col_B = c * 2 + j_prime
                        V[row_B * 2 + col_B, col] += coeff_outer * coeff_inner
            
            # W contributions
            # C is 4x2.
            # Block (r, c) of C covers rows 2r..2r+1, cols 2c..2c+1.
            # Here C has only 1 col of blocks (c=0).
            for i_prime in range(2): # row in block
                for j_prime in range(2): # col in block
                    coeff_inner = W2[j_prime * 2 + i_prime, l]
                    if coeff_inner == 0: continue
                    for (r, c), coeff_outer in C_coeffs[k].items():
                        row_C = r * 2 + i_prime
                        col_C = c * 2 + j_prime
                        # W index: k*n + i. n=4 (rows of C).
                        # W[col_C * 4 + row_C, col]
                        W[col_C * 4 + row_C, col] += coeff_outer * coeff_inner
                        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # 4x4x4 part
    U1, V1, W1 = construct_444()
    # 4x4x2 part
    U2, V2, W2 = construct_442()
    
    # Concatenate
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Convert to lists for output
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Quick verification
    np.random.seed(42)
    A = np.random.randint(-5, 5, (4, 4))
    B = np.random.randint(-5, 5, (4, 6))
    
    C_expected = A @ B
    
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    rank = U.shape[1]
    
    # Compute C via tensor decomposition
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is O(n*m*p*rank). 4*4*6*77 = 7392 ops. Fast.
    C_calc = np.zeros((4, 6))
    for i in range(4):
        for k in range(6):
            s = 0
            for j in range(4):
                for r in range(rank):
                    s += U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
            C_calc[i, k] = s
            
    if np.allclose(C_expected, C_calc):
        print("Verification Passed")
    else:
        print("Verification Failed")
        print(C_expected)
        print(C_calc)
