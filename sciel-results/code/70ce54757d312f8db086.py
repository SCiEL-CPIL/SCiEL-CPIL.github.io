"""Matrix multiplication 4x4x5 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 Strassen (rank 49)
    # Part 2: 4x4x1 schoolbook (rank 16)
    # Total rank: 65
    rank = 65
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # 2x2 Strassen: returns 7 terms, each term is (u_dict, v_dict, w_dict)
    # where u_dict maps (i,j) -> coeff, etc.
    # For a 2x2 product Z = X*Y:
    # X indexed by (i,j), Y by (j,k), Z by (i,k)
    # In our flattened indices: X[i*2+j], Y[j*2+k], Z[k*2+i]
    
    def get_strassen_2x2():
        """Return 7 terms for 2x2 Strassen.
        Each term: (x_coeffs, y_coeffs, z_coeffs)
        x_coeffs: dict (i,j) -> val
        y_coeffs: dict (j,k) -> val
        z_coeffs: dict (i,k) -> val
        """
        terms = []
        # T0: (X00+X11)(Y00+Y11) -> Z00 + Z11
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # T1: (X10+X11)(Y00) -> Z10
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # T2: (X00)(Y01-Y11) -> Z01
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}))
        # T3: (X11)(Y10-Y00) -> Z10 + Z11
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (1,1):1}))
        # T4: (X00+X01)(Y11) -> Z01 + Z11
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}))
        # T5: (X10-X00)(Y00+Y01) -> Z10
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1}))
        # T6: (X01-X11)(Y10+Y11) -> Z11
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}))
        return terms
    
    strassen_terms = get_strassen_2x2()
    
    # 4x4 block Strassen:
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    # C blocks: C00=C[0:2,0:2], C01=C[0:2,2:4], C10=C[2:4,0:2], C11=C[2:4,2:4]
    
    # Block Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M1 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 + M6
    
    # For each block product M = X*Y, we expand using 2x2 Strassen.
    # X is a 2x2 submatrix of A, Y is a 2x2 submatrix of B.
    # The result M contributes to C blocks with certain signs.
    
    # Let's define the 7 block products and their contributions to C blocks:
    # Each block product is: X (2x2 from A) * Y (2x2 from B) -> M (2x2)
    # M then contributes to C blocks.
    
    # For block product s (s=0..6):
    # X_s is a linear combination of A blocks
    # Y_s is a linear combination of B blocks
    # M_s contributes to C blocks with certain coefficients
    
    block_products = [
        # (x_blocks, y_blocks, c_block_contribs)
        # x_blocks: dict (block_row, block_col) -> coeff for A
        # y_blocks: dict (block_row, block_col) -> coeff for B
        # c_block_contribs: dict (block_row, block_col) -> coeff for C
        #
        # A blocks: (0,0)=A00, (0,1)=A01, (1,0)=A10, (1,1)=A11
        # B blocks: (0,0)=B00, (0,1)=B01, (1,0)=B10, (1,1)=B11
        # C blocks: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M0
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):-1, (0,0):-1}),              # M1 (C10+=M1, C00-=M1)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),              # M2 (C01+=M2, C11-=M2)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1, (1,1):1}),      # M3 (C10+=M3, C00+=M3, C11+=M3)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),                # M4 (C01+=M4, C11+=M4)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,0):1}),      # M5 (C10+=M5, C00+=M5)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),               # M6 (C11+=M6)
    ]
    
    # Wait, I need to double-check the C block contributions.
    # C00 = M0 + M3 - M1 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 + M6
    
    # So:
    # M0: C00 += 1, C11 += 1
    # M1: C00 += -1, C10 += 1
    # M2: C01 += 1, C11 += -1
    # M3: C00 += 1, C10 += 1, C11 += 1
    # M4: C01 += 1, C11 += 1
    # M5: C00 += 1, C10 += 1
    # M6: C11 += 1
    
    block_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1, (1,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,0):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),
    ]
    
    col = 0
    for bp_idx in range(7):
        x_blocks, y_blocks, c_contribs = block_products[bp_idx]
        
        # For each of the 7 Strassen terms within this block product:
        for s_idx in range(7):
            x_coeffs, y_coeffs, z_coeffs = strassen_terms[s_idx]
            
            # x_coeffs: dict (i,j) -> val, where (i,j) is within the 2x2 block
            # We need to map this to the full 4x4 A matrix.
            # x_blocks tells us which A blocks are involved and with what coefficient.
            
            # Build U entries for this term:
            u_entries = {}  # (i,j) in 4x4 -> coeff
            for (br, bc), bc_coeff in x_blocks.items():
                for (i, j), x_coeff in x_coeffs.items():
                    full_i = br * 2 + i
                    full_j = bc * 2 + j
                    key = (full_i, full_j)
                    u_entries[key] = u_entries.get(key, 0) + bc_coeff * x_coeff
            
            # Build V entries:
            v_entries = {}  # (j,k) in 4x4 (but k only 0..3 for this part) -> coeff
            for (br, bc), bc_coeff in y_blocks.items():
                for (j, k), y_coeff in y_coeffs.items():
                    full_j = br * 2 + j
                    full_k = bc * 2 + k
                    key = (full_j, full_k)
                    v_entries[key] = v_entries.get(key, 0) + bc_coeff * y_coeff
            
            # Build W entries:
            # z_coeffs: dict (i,k) -> val, where (i,k) is within the 2x2 result block
            # c_contribs tells us which C blocks this M contributes to and with what coefficient.
            w_entries = {}  # (i,k) in 4x4 (k only 0..3) -> coeff
            for (br, bc), cc_coeff in c_contribs.items():
                for (i, k), z_coeff in z_coeffs.items():
                    full_i = br * 2 + i
                    full_k = bc * 2 + k
                    key = (full_i, full_k)
                    w_entries[key] = w_entries.get(key, 0) + cc_coeff * z_coeff
            
            # Add to U, V, W:
            for (i, j), val in u_entries.items():
                if val != 0:
                    U[i*m + j][col] += val
            for (j, k), val in v_entries.items():
                if val != 0:
                    V[j*p + k][col] += val
            for (i, k), val in w_entries.items():
                if val != 0:
                    W[k*n + i][col] += val
            
            col += 1
    
    # Part 2: 4x4x1 schoolbook for column k=4
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each (i,j): U[i*4+j]=1, V[j*5+4]=1, W[4*4+i]=1
    for i in range(4):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
