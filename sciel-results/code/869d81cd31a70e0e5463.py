import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4 x 4x4 -> 4x4 (rank 49 via Strassen)
    # Part 2: 4x4 x 4x3 -> 4x3 (rank 48 via schoolbook)
    # Total rank = 49 + 48 = 97
    
    R = 97
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 (B cols 0-3) ---
    # Strassen products for 2x2 block matrices:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 - P4 + P5 - P6
    # C12 = P3 + P6
    # C21 = P2 + P7
    # C22 = P1 - P3 + P4 + P7
    #
    # Each Pi is a 2x2 matrix product. For each Pi, we need to decompose
    # the 2x2x2 tensor into 7 rank-1 terms.
    #
    # The 7 rank-1 terms for 2x2x2 Strassen:
    # For X(2x2) * Y(2x2) -> Z(2x2):
    # s1 = (x00+x11)(y00+y11) -> z00+=s1, z11+=s1, z01-=s1, z10-=s1
    # s2 = (x10+x11)y00 -> z10+=s2, z11+=s2, z00-=s2
    # s3 = x00(y01-y11) -> z01+=s3, z00-=s3
    # s4 = x11(y10-y00) -> z10+=s4, z11-=s4
    # s5 = (x00+x01)y11 -> z01+=s5, z11+=s5, z10-=s5
    # s6 = (x10-x00)(y00+y01) -> z10+=s6, z01+=s6, z00-=s6
    # s7 = (x01-x11)(y10+y11) -> z11+=s7, z01-=s7, z10+=s7
    
    # Let me define a helper to add a 2x2x2 Strassen decomposition
    # given the linear forms for X and Y blocks.
    
    def add_2x2_strassen(a_forms, b_forms, c_offset, c_signs):
        """
        a_forms: list of 4 (coeff, i, j) for the 2x2 A block
        b_forms: list of 4 (coeff, j, k) for the 2x2 B block  
        c_offset: base index for C block in W
        c_signs: 2x2 list of signs for how this contributes to C
        """
        nonlocal col
        # The 7 Strassen scalar products for 2x2x2
        # Each term: (linear form in A) * (linear form in B) contributes to C entries
        # with specific coefficients.
        
        # For a 2x2 product X*Y=Z:
        # Z[0,0] = x00*y00 + x01*y10
        # Z[0,1] = x00*y01 + x01*y11
        # Z[1,0] = x10*y00 + x11*y10
        # Z[1,1] = x10*y01 + x11*y11
        
        # Strassen:
        # s1 = (x00+x11)(y00+y11)
        # s2 = (x10+x11)y00
        # s3 = x00(y01-y11)
        # s4 = x11(y10-y00)
        # s5 = (x00+x01)y11
        # s6 = (x10-x00)(y00+y01)
        # s7 = (x01-x11)(y10+y11)
        
        # Z00 = s1 - s4 + s5 - s6
        # Z01 = s3 + s6
        # Z10 = s2 + s7
        # Z11 = s1 - s3 + s4 + s7
        
        # For each of the 7 scalar products, we add 4 rank-1 terms? No.
        # Each scalar product s_r contributes to multiple Z entries with coefficients.
        # But in the tensor decomposition, each rank-1 term is a single product.
        # The 7 products ARE the rank-1 terms for the 2x2x2 tensor.
        
        # So for each s_r, we have:
        # U contribution: the linear form in A
        # V contribution: the linear form in B
        # W contribution: for each C entry that s_r contributes to, with the coefficient
        
        # s1: A-form = x00+x11, B-form = y00+y11
        #   Z00 += s1, Z11 += s1, Z01 -= s1, Z10 -= s1
        # s2: A-form = x10+x11, B-form = y00
        #   Z10 += s2, Z11 += s2, Z00 -= s2
        # s3: A-form = x00, B-form = y01-y11
        #   Z01 += s3, Z00 -= s3
        # s4: A-form = x11, B-form = y10-y00
        #   Z10 += s4, Z11 -= s4
        # s5: A-form = x00+x01, B-form = y11
        #   Z01 += s5, Z11 += s5, Z10 -= s5
        # s6: A-form = x10-x00, B-form = y00+y01
        #   Z10 += s6, Z01 += s6, Z00 -= s6
        # s7: A-form = x01-x11, B-form = y10+y11
        #   Z11 += s7, Z01 -= s7, Z10 += s7
        
        # Now, x00, x01, x10, x11 are linear forms in A (from a_forms)
        # y00, y01, y10, y11 are linear forms in B (from b_forms)
        
        # Let me represent linear forms as dicts: {(i,j): coeff}
        def add_form(target, source, scale=1):
            for (idx, c) in source.items():
                target[idx] = target.get(idx, 0) + c * scale
        
        # a_forms is a dict: {(0,0): form00, (0,1): form01, (1,0): form10, (1,1): form11}
        # where each form is a dict {A_index: coeff}
        # Similarly for b_forms with B_index
        
        # Define the 7 products' A and B linear forms
        # x00 = a_forms[(0,0)], x01 = a_forms[(0,1)], etc.
        
        products = []
        
        # s1: (x00+x11) * (y00+y11)
        a1 = {}
        add_form(a1, a_forms[(0,0)], 1)
        add_form(a1, a_forms[(1,1)], 1)
        b1 = {}
        add_form(b1, b_forms[(0,0)], 1)
        add_form(b1, b_forms[(1,1)], 1)
        c1 = {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}
        products.append((a1, b1, c1))
        
        # s2: (x10+x11) * y00
        a2 = {}
        add_form(a2, a_forms[(1,0)], 1)
        add_form(a2, a_forms[(1,1)], 1)
        b2 = dict(b_forms[(0,0)])
        c2 = {(1,0): 1, (1,1): 1, (0,0): -1}
        products.append((a2, b2, c2))
        
        # s3: x00 * (y01-y11)
        a3 = dict(a_forms[(0,0)])
        b3 = {}
        add_form(b3, b_forms[(0,1)], 1)
        add_form(b3, b_forms[(1,1)], -1)
        c3 = {(0,1): 1, (0,0): -1}
        products.append((a3, b3, c3))
        
        # s4: x11 * (y10-y00)
        a4 = dict(a_forms[(1,1)])
        b4 = {}
        add_form(b4, b_forms[(1,0)], 1)
        add_form(b4, b_forms[(0,0)], -1)
        c4 = {(1,0): 1, (1,1): -1}
        products.append((a4, b4, c4))
        
        # s5: (x00+x01) * y11
        a5 = {}
        add_form(a5, a_forms[(0,0)], 1)
        add_form(a5, a_forms[(0,1)], 1)
        b5 = dict(b_forms[(1,1)])
        c5 = {(0,1): 1, (1,1): 1, (1,0): -1}
        products.append((a5, b5, c5))
        
        # s6: (x10-x00) * (y00+y01)
        a6 = {}
        add_form(a6, a_forms[(1,0)], 1)
        add_form(a6, a_forms[(0,0)], -1)
        b6 = {}
        add_form(b6, b_forms[(0,0)], 1)
        add_form(b6, b_forms[(0,1)], 1)
        c6 = {(1,0): 1, (0,1): 1, (0,0): -1}
        products.append((a6, b6, c6))
        
        # s7: (x01-x11) * (y10+y11)
        a7 = {}
        add_form(a7, a_forms[(0,1)], 1)
        add_form(a7, a_forms[(1,1)], -1)
        b7 = {}
        add_form(b7, b_forms[(1,0)], 1)
        add_form(b7, b_forms[(1,1)], 1)
        c7 = {(1,1): 1, (0,1): -1, (1,0): 1}
        products.append((a7, b7, c7))
        
        # Now add each product as a rank-1 term
        for a_form, b_form, c_form in products:
            # U: A has shape 4x4, index i*m+j = i*4+j
            for (ai, aj), ac in a_form.items():
                U[ai * m + aj][col] += ac
            
            # V: B has shape 4x4, index j*p+k = j*7+k
            for (bj, bk), bc in b_form.items():
                V[bj * p + bk][col] += bc
            
            # W: C has shape 4x4, index k*n+i = k*4+i
            # c_form keys are (z_row, z_col) where z_row is i-block, z_col is k-block
            # Within the 2x2 block, z[ri, ci] maps to global C[2*br+ri, 2*bc+ci]
            # c_offset gives (br, bc)
            br, bc = c_offset
            for (zi, zk), cc in c_form.items():
                global_i = 2 * br + zi
                global_k = 2 * bc + zk
                W[global_k * n + global_i][col] += cc
            
            col += 1
    
    # Define the 7 Strassen products at the block level
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # Each block is 2x2, so A11[i,j] for i,j in {0,1} corresponds to global A[2*0+i, 2*0+j] = A[i,j]
    
    # P1 = (A11+A22)(B11+B22)
    a_p1 = {
        (0,0): {(0,0): 1, (2,2): 1},   # x00 = A[0,0]+A[2,2]
        (0,1): {(0,1): 1, (2,3): 1},   # x01 = A[0,1]+A[2,3]
        (1,0): {(1,0): 1, (3,2): 1},   # x10 = A[1,0]+A[3,2]
        (1,1): {(1,1): 1, (3,3): 1},   # x11 = A[1,1]+A[3,3]
    }
    b_p1 = {
        (0,0): {(0,0): 1, (2,2): 1},   # y00 = B[0,0]+B[2,2]
        (0,1): {(0,1): 1, (2,3): 1},   # y01 = B[0,1]+B[2,3]
        (1,0): {(1,0): 1, (3,2): 1},   # y10 = B[1,0]+B[3,2]
        (1,1): {(1,1): 1, (3,3): 1},   # y11 = B[1,1]+B[3,3]
    }
    add_2x2_strassen(a_p1, b_p1, (0,0), None)
    
    # P2 = (A21+A22)B11
    a_p2 = {
        (0,0): {(1,0): 1, (3,2): 1},   # x00 = A[1,0]+A[3,2]
        (0,1): {(1,1): 1, (3,3): 1},   # x01 = A[1,1]+A[3,3]
        (1,0): {(1,0): 1, (3,2): 1},   # x10 = A[1,0]+A[3,2]
        (1,1): {(1,1): 1, (3,3): 1},   # x11 = A[1,1]+A[3,3]
    }
    b_p2 = {
        (0,0): {(0,0): 1},   # y00 = B[0,0]
        (0,1): {(0,1): 1},   # y01 = B[0,1]
        (1,0): {(1,0): 1},   # y10 = B[1,0]
        (1,1): {(1,1): 1},   # y11 = B[1,1]
    }
    add_2x2_strassen(a_p2, b_p2, (1,0), None)
    
    # P3 = A11(B12-B22)
    a_p3 = {
        (0,0): {(0,0): 1},   # x00 = A[0,0]
        (0,1): {(0,1): 1},   # x01 = A[0,1]
        (1,0): {(1,0): 1},   # x10 = A[1,0]
        (1,1): {(1,1): 1},   # x11 = A[1,1]
    }
    b_p3 = {
        (0,0): {(0,2): 1, (2,2): -1},   # y00 = B[0,2]-B[2,2]
        (0,1): {(0,3): 1, (2,3): -1},   # y01 = B[0,3]-B[2,3]
        (1,0): {(1,2): 1, (3,2): -1},   # y10 = B[1,2]-B[3,2]
        (1,1): {(1,3): 1, (3,3): -1},   # y11 = B[1,3]-B[3,3]
    }
    add_2x2_strassen(a_p3, b_p3, (0,1), None)
    
    # P4 = A22(B21-B11)
    a_p4 = {
        (0,0): {(2,0): 1},   # x00 = A[2,0]
        (0,1): {(2,1): 1},   # x01 = A[2,1]
        (1,0): {(3,0): 1},   # x10 = A[3,0]
        (1,1): {(3,1): 1},   # x11 = A[3,1]
    }
    b_p4 = {
        (0,0): {(2,0): 1, (0,0): -1},   # y00 = B[2,0]-B[0,0]
        (0,1): {(2,1): 1, (0,1): -1},   # y01 = B[2,1]-B[0,1]
        (1,0): {(3,0): 1, (1,0): -1},   # y10 = B[3,0]-B[1,0]
        (1,1): {(3,1): 1, (1,1): -1},   # y11 = B[3,1]-B[1,1]
    }
    add_2x2_strassen(a_p4, b_p4, (1,1), None)
    
    # P5 = (A11+A12)B22
    a_p5 = {
        (0,0): {(0,0): 1, (0,2): 1},   # x00 = A[0,0]+A[0,2]
        (0,1): {(0,1): 1, (0,3): 1},   # x01 = A[0,1]+A[0,3]
        (1,0): {(1,0): 1, (1,2): 1},   # x10 = A[1,0]+A[1,2]
        (1,1): {(1,1): 1, (1,3): 1},   # x11 = A[1,1]+A[1,3]
    }
    b_p5 = {
        (0,0): {(2,0): 1},   # y00 = B[2,0]
        (0,1): {(2,1): 1},   # y01 = B[2,1]
        (1,0): {(3,0): 1},   # y10 = B[3,0]
        (1,1): {(3,1): 1},   # y11 = B[3,1]
    }
    add_2x2_strassen(a_p5, b_p5, (0,1), None)
    
    # P6 = (A21-A11)(B11+B12)
    a_p6 = {
        (0,0): {(1,0): 1, (0,0): -1},   # x00 = A[1,0]-A[0,0]
        (0,1): {(1,1): 1, (0,1): -1},   # x01 = A[1,1]-A[0,1]
        (1,0): {(1,0): 1, (0,0): -1},   # x10 = A[1,0]-A[0,0]
        (1,1): {(1,1): 1, (0,1): -1},   # x11 = A[1,1]-A[0,1]
    }
    b_p6 = {
        (0,0): {(0,0): 1, (0,2): 1},   # y00 = B[0,0]+B[0,2]
        (0,1): {(0,1): 1, (0,3): 1},   # y01 = B[0,1]+B[0,3]
        (1,0): {(1,0): 1, (1,2): 1},   # y10 = B[1,0]+B[1,2]
        (1,1): {(1,1): 1, (1,3): 1},   # y11 = B[1,1]+B[1,3]
    }
    add_2x2_strassen(a_p6, b_p6, (0,0), None)
    
    # P7 = (A12-A22)(B21+B22)
    a_p7 = {
        (0,0): {(0,2): 1, (2,0): -1},   # x00 = A[0,2]-A[2,0]
        (0,1): {(0,3): 1, (2,1): -1},   # x01 = A[0,3]-A[2,1]
        (1,0): {(1,2): 1, (3,0): -1},   # x10 = A[1,2]-A[3,0]
        (1,1): {(1,3): 1, (3,1): -1},   # x11 = A[1,3]-A[3,1]
    }
    b_p7 = {
        (0,0): {(2,0): 1, (2,2): 1},   # y00 = B[2,0]+B[2,2]
        (0,1): {(2,1): 1, (2,3): 1},   # y01 = B[2,1]+B[2,3]
        (1,0): {(3,0): 1, (3,2): 1},   # y10 = B[3,0]+B[3,2]
        (1,1): {(3,1): 1, (3,3): 1},   # y11 = B[3,1]+B[3,3]
    }
    add_2x2_strassen(a_p7, b_p7, (1,1), None)
    
    # --- Part 2: Schoolbook for 4x4x3 (B cols 4-6) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == R, f"Expected {R} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
