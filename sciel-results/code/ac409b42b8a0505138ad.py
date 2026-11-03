"""Program for matrix multiplication tensor decomposition 4x4 * 4x8."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Return rank-minimal tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses nested Strassen decomposition.
    Rank 49 for 4x4x4, used twice for 4x4x8. Total rank 98.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W uses convention W[k*n + i, r] for C[i,k].
    """
    n, m, p = DIMENSIONS
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Define 2x2 Strassen decomposition
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
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
    
    # We define the 7 terms of the 2x2 Strassen decomposition.
    # Each term r corresponds to a product x_r * y_r.
    # x_r is a linear combination of A elements.
    # y_r is a linear combination of B elements.
    # The result contributes to C elements via W2.
    
    # Term 0 (M1): x = A00+A11, y = B00+B11
    # A indices: 00 (0), 11 (3). B indices: 00 (0), 11 (3).
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # C contributions: C00 += 1, C11 += 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 is index 0 (k=0,i=0), C11 is index 3 (k=1,i=1)
    
    # Term 1 (M2): x = A10+A11, y = B00
    # A indices: 10 (2), 11 (3). B indices: 00 (0).
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # C contributions: C10 += 1, C11 -= 1
    W2[2][1] = 1; W2[3][1] = -1 # C10 is index 2 (k=0,i=1), C11 is index 3
    
    # Term 2 (M3): x = A00, y = B01-B11
    # A indices: 00 (0). B indices: 01 (1), 11 (3).
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # C contributions: C01 += 1, C11 += 1
    W2[1][2] = 1; W2[3][2] = 1 # C01 is index 1 (k=1,i=0), C11 is index 3
    
    # Term 3 (M4): x = A11, y = B10-B00
    # A indices: 11 (3). B indices: 10 (2), 00 (0).
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    # C contributions: C00 += 1, C10 += 1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 4 (M5): x = A00+A01, y = B11
    # A indices: 00 (0), 01 (1). B indices: 11 (3).
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # C contributions: C00 -= 1, C01 += 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 5 (M6): x = A10-A00, y = B00+B01
    # A indices: 10 (2), 00 (0). B indices: 00 (0), 01 (1).
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # C contributions: C11 += 1
    W2[3][5] = 1
    
    # Term 6 (M7): x = A01-A11, y = B10+B11
    # A indices: 01 (1), 11 (3). B indices: 10 (2), 11 (3).
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # C contributions: C00 += 1
    W2[0][6] = 1

    # 2. Construct 4x4x4 decomposition using nested Strassen
    # Block level Strassen terms:
    # M1_b = (A00+A11) * (B00+B11)
    # M2_b = (A10+A11) * B00
    # M3_b = A00 * (B01-B11)
    # M4_b = A11 * (B10-B00)
    # M5_b = (A00+A01) * B11
    # M6_b = (A10-A00) * (B00+B01)
    # M7_b = (A01-A11) * (B10+B11)
    
    # C blocks:
    # C00_b = M1_b + M4_b - M5_b + M7_b
    # C01_b = M3_b + M5_b
    # C10_b = M2_b + M4_b
    # C11_b = M1_b - M2_b + M3_b + M6_b
    
    # We iterate over the 7 block terms.
    # For each block term, we have X and Y which are 2x2 matrices.
    # We apply the 2x2 Strassen decomposition (U2, V2, W2) to X*Y.
    # This means we iterate over the 7 inner terms of U2, V2, W2.
    
    # We need to map the "elements" of X and Y to the original A and B.
    # Let's define the linear forms for X and Y for each block term.
    
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    
    # Block Term 0: X = A00+A11, Y = B00+B11
    # X elements:
    # X00 = A00[0]+A11[0], X01 = A00[1]+A11[1], X10 = A00[2]+A11[2], X11 = A00[3]+A11[3]
    # Y elements:
    # Y00 = B00[0]+B11[0], ...
    
    # This is getting complicated to hardcode the mappings.
    # Let's write a helper that takes the linear forms and adds to U, V, W.
    
    def add_block_term(x_coeffs, y_coeffs, c_coeffs, col_offset):
        # x_coeffs: list of 4 lists of (index, coeff) for A
        # y_coeffs: list of 4 lists of (index, coeff) for B
        # c_coeffs: list of 4 lists of (index, coeff) for C
        # We apply 2x2 Strassen to X*Y.
        
        for r_inner in range(7):
            col = col_offset + r_inner
            
            # U contribution: sum over inner Strassen U2 elements
            # U2[i*2+j, r_inner] is the coefficient for X[i,j]
            # X[i,j] is a linear form in A.
            for i in range(2):
                for j in range(2):
                    coeff_x = U2[i*2+j][r_inner]
                    if coeff_x != 0:
                        # Add coeff_x * X[i,j] to U
                        for idx_a, c_a in x_coeffs[i*2+j]:
                            U[idx_a][col] += coeff_x * c_a
                            
            # V contribution: sum over inner Strassen V2 elements
            # V2[j*2+k, r_inner] is the coefficient for Y[j,k]
            # Y[j,k] is a linear form in B.
            for j in range(2):
                for k in range(2):
                    coeff_y = V2[j*2+k][r_inner]
                    if coeff_y != 0:
                        for idx_b, c_b in y_coeffs[j*2+k]:
                            V[idx_b][col] += coeff_y * c_b
                            
            # W contribution: sum over inner Strassen W2 elements
            # W2[k*2+i, r_inner] is the coefficient for Z[k,i]
            # Z is the result of X*Y.
            # The block term result contributes to C blocks via c_coeffs.
            # Z is a 2x2 matrix.
            # Z[k,i] contributes to C blocks.
            for k in range(2):
                for i in range(2):
                    coeff_z = W2[k*2+i][r_inner]
                    if coeff_z != 0:
                        for idx_c, c_c in c_coeffs[k*2+i]:
                            W[idx_c][col] += coeff_z * c_c

    # Define x_coeffs, y_coeffs, c_coeffs for each of the 7 block terms.
    # A indices: 0..15. B indices: 0..15. C indices: 0..15.
    
    # Helper to get block indices
    def block_indices(block_r, block_c):
        # block_r, block_c in {0,1}
        base = (block_r * 2 + block_c) * 4
        return [base, base+1, base+2, base+3]
    
    # Block Term 0: X = A00+A11, Y = B00+B11
    # X00 = A00[0]+A11[0], etc.
    x0 = []
    for i in range(4):
        x0.append([(block_indices(0,0)[i], 1), (block_indices(1,1)[i], 1)])
    y0 = []
    for i in range(4):
        y0.append([(block_indices(0,0)[i], 1), (block_indices(1,1)[i], 1)])
    c0 = []
    # C00_b += 1, C11_b += 1
    for i in range(4):
        c0.append([(block_indices(0,0)[i], 1), (block_indices(1,1)[i], 1)])
    add_block_term(x0, y0, c0, 0)
    
    # Block Term 1: X = A10+A11, Y = B00
    x1 = []
    for i in range(4):
        x1.append([(block_indices(1,0)[i], 1), (block_indices(1,1)[i], 1)])
    y1 = []
    for i in range(4):
        y1.append([(block_indices(0,0)[i], 1)])
    c1 = []
    # C10_b += 1, C11_b -= 1
    for i in range(4):
        c1.append([(block_indices(1,0)[i], 1), (block_indices(1,1)[i], -1)])
    add_block_term(x1, y1, c1, 7)
    
    # Block Term 2: X = A00, Y = B01-B11
    x2 = []
    for i in range(4):
        x2.append([(block_indices(0,0)[i], 1)])
    y2 = []
    for i in range(4):
        y2.append([(block_indices(0,1)[i], 1), (block_indices(1,1)[i], -1)])
    c2 = []
    # C01_b += 1, C11_b += 1
    for i in range(4):
        c2.append([(block_indices(0,1)[i], 1), (block_indices(1,1)[i], 1)])
    add_block_term(x2, y2, c2, 14)
    
    # Block Term 3: X = A11, Y = B10-B00
    x3 = []
    for i in range(4):
        x3.append([(block_indices(1,1)[i], 1)])
    y3 = []
    for i in range(4):
        y3.append([(block_indices(1,0)[i], 1), (block_indices(0,0)[i], -1)])
    c3 = []
    # C00_b += 1, C10_b += 1
    for i in range(4):
        c3.append([(block_indices(0,0)[i], 1), (block_indices(1,0)[i], 1)])
    add_block_term(x3, y3, c3, 21)
    
    # Block Term 4: X = A00+A01, Y = B11
    x4 = []
    for i in range(4):
        x4.append([(block_indices(0,0)[i], 1), (block_indices(0,1)[i], 1)])
    y4 = []
    for i in range(4):
        y4.append([(block_indices(1,1)[i], 1)])
    c4 = []
    # C00_b -= 1, C01_b += 1
    for i in range(4):
        c4.append([(block_indices(0,0)[i], -1), (block_indices(0,1)[i], 1)])
    add_block_term(x4, y4, c4, 28)
    
    # Block Term 5: X = A10-A00, Y = B00+B01
    x5 = []
    for i in range(4):
        x5.append([(block_indices(1,0)[i], 1), (block_indices(0,0)[i], -1)])
    y5 = []
    for i in range(4):
        y5.append([(block_indices(0,0)[i], 1), (block_indices(0,1)[i], 1)])
    c5 = []
    # C11_b += 1
    for i in range(4):
        c5.append([(block_indices(1,1)[i], 1)])
    add_block_term(x5, y5, c5, 35)
    
    # Block Term 6: X = A01-A11, Y = B10+B11
    x6 = []
    for i in range(4):
        x6.append([(block_indices(0,1)[i], 1), (block_indices(1,1)[i], -1)])
    y6 = []
    for i in range(4):
        y6.append([(block_indices(1,0)[i], 1), (block_indices(1,1)[i], 1)])
    c6 = []
    # C00_b += 1
    for i in range(4):
        c6.append([(block_indices(0,0)[i], 1)])
    add_block_term(x6, y6, c6, 42)
    
    # Now we have the 4x4x4 decomposition in columns 0..48.
    # We need to duplicate it for the second half of B (B1, B2).
    # The second half uses B indices 16..31 (since B is 4x8, m=4, p=8).
    # Wait, B is 4x8. m=4, p=8.
    # B indices: 0..31.
    # B1 is cols 0..3 (indices 0..15).
    # B2 is cols 4..7 (indices 16..31).
    # The U matrix is the same for both.
    # The V matrix for the second half should use B2 indices.
    # The W matrix for the second half should use C2 indices.
    # C is 4x8. n=4, p=8.
    # C1 is cols 0..3 (indices 0..15).
    # C2 is cols 4..7 (indices 16..31).
    
    # We copy U from 0..48 to 49..97.
    for i in range(n*m):
        for r in range(49):
            U[i][49+r] = U[i][r]
            
    # We construct V for the second half.
    # V for first half used B indices 0..15.
    # V for second half should use B indices 16..31.
    # We can just shift the indices by 16.
    for i in range(m*p): # m*p = 32
        for r in range(49):
            # V[i][r] is non-zero only if i < 16 (first half of B)
            # If i >= 16, it's 0 in the first half decomposition.
            # So we can just copy V[i][r] to V[i][49+r] if i < 16?
            # No, V[i][r] for i in 0..15 corresponds to B1.
            # For the second half, we want V[i+16][49+r] to be V[i][r].
            # And V[i][49+r] should be 0 for i in 0..15.
            # And V[i+16][r] should be 0 for i in 0..15.
            
            pass
            
    # Easier: Re-run add_block_term with shifted indices for V and W.
    # Or just copy and shift.
    
    # Let's just copy the logic.
    # For r in 0..48:
    #   U[i][49+r] = U[i][r]
    #   V[i][49+r] = V[i-16][r] if i >= 16 else 0
    #   W[i][49+r] = W[i-16][r] if i >= 16 else 0
    
    for i in range(n*m):
        for r in range(49):
            U[i][49+r] = U[i][r]
            
    for i in range(m*p):
        for r in range(49):
            if i >= 16:
                V[i][49+r] = V[i-16][r]
            else:
                V[i][49+r] = 0
                
    for i in range(p*n):
        for r in range(49):
            if i >= 16:
                W[i][49+r] = W[i-16][r]
            else:
                W[i][49+r] = 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
