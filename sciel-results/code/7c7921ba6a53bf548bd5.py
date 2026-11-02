"""Block Strassen decomposition for 4x4*4x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    U_list = []
    V_list = []
    W_list = []

    def add_mult(u_val, v_val, w_val, a_idx, b_idx, c_idx):
        """Add one scalar multiplication: C[c] += a * b * u * v * w pattern."""
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        U_row[a_idx] = u_val
        V_row[b_idx] = v_val
        W_row[c_idx] = w_val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def add_strassen_2x2x2(a_rows, b_cols, c_base):
        """
        Compute C[c_base:c_base+2, :] += A[a_rows, :] * B[:, b_cols]
        where A is 2x4, B is 4x2.
        a_rows: list of 2 row indices of A (for the 2x2 block of A's columns split)
        b_cols: list of 2 col indices of B
        c_base: starting index of the 2 output columns in C (within the 2x7 block)
        
        A block: 2x4, we split into A[0:2, 0:2] and A[0:2, 2:4]
        B block: 4x2, we split into B[0:2, :] and B[2:4, :]
        
        C_2x2 = A_left * B_top + A_right * B_bottom
        where A_left = A[rows, cols 0:2], A_right = A[rows, cols 2:4]
        B_top = B[cols 0:2, b_cols], B_bottom = B[cols 2:4, b_cols]
        
        Strassen on each 2x2x2: rank 7 each, total 14.
        """
        r0, r1 = a_rows  # the 2 rows of A
        bc0, bc1 = b_cols  # the 2 columns of B
        # A rows r0,r1; B cols bc0,bc1
        # A is 2x4: rows r0,r1, cols 0..3
        # B is 4x2: rows 0..3, cols bc0,bc1
        
        # Split A into A11=A[r0,r1, 0:2], A12=A[r0,r1, 2:4]
        # Split B into B11=B[0:2, bc0,bc1], B12=B[2:4, bc0,bc1]
        
        # Strassen for A11 * B11 (2x2 * 2x2 -> 2x2)
        # A11 = [[a00, a01], [a10, a11]] where a_ij = A[r_i, j]
        # B11 = [[b00, b01], [b10, b11]] where b_ij = B[i, bc_j]
        
        a00 = (r0, 0); a01 = (r0, 1); a10 = (r1, 0); a11 = (r1, 1)
        b00 = (0, bc0); b01 = (0, bc1); b10 = (2, bc0); b11 = (2, bc1)
        
        # M1 = (a10+a11)*(b00+b01)
        add_mult(1, 1, 1, a10[0]*m+a10[1], b10[0]*p+b10[1], c_base*0+n*0+r0)  # placeholder
        # Actually let me redo this more carefully with explicit index tracking.
        pass

    # Let me take a cleaner approach: build the full decomposition explicitly.
    # Reset
    U_list = []
    V_list = []
    W_list = []

    def add_entry(a_row, a_col, b_row, b_col, c_row, c_col, coeff=1.0):
        """Add one rank-1 term: U[a_idx]=coeff, V[b_idx]=1, W[c_idx]=1"""
        a_idx = a_row * m + a_col
        b_idx = b_row * p + b_col
        c_idx = c_row * n + c_col
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        U_row[a_idx] = coeff
        V_row[b_idx] = 1.0
        W_row[c_idx] = 1.0
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def add_entry2(a_row, a_col, b_row, b_col, c_row, c_col, coeff_u, coeff_v, coeff_w):
        """More flexible: U[a_idx]=coeff_u, V[b_idx]=coeff_v, W[c_idx]=coeff_w"""
        a_idx = a_row * m + a_col
        b_idx = b_row * p + b_col
        c_idx = c_row * n + c_col
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        U_row[a_idx] = coeff_u
        V_row[b_idx] = coeff_v
        W_row[c_idx] = coeff_w
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    # For Strassen 2x2x2:
    # X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
    # M1=(x10+x11)(y00+y01), M2=(x10+x11)(y10+y11), M3=(x00-x10)(y00+y10)
    # M4=(x01-x11)(y01+y11), M5=(x00+x01)(y11-y01), M6=(x00+x11)(y00-y10), M7=(x01+x11)(y10-y00)
    # Z00=M5+M7-M2+M4, Z01=M1+M2, Z10=M3+M4, Z11=M1-M3+M5+M6
    
    # In our tensor decomposition, each multiplication M_j contributes to U, V, W.
    # U has entries for A's elements, V for B's, W for C's.
    # For a product term like (x_a + x_b) * (y_c + y_d):
    # This expands to 4 scalar multiplications, but in Strassen it's ONE multiplication
    # where U has 2 non-zero entries (coefficients of x_a and x_b) and V has 2 non-zero entries.
    
    # So for M1 = (x10+x11)*(y00+y01):
    # U has coeff 1 at x10's position and coeff 1 at x11's position
    # V has coeff 1 at y00's position and coeff 1 at y01's position
    # W has coeff 1 at the appropriate Z positions
    
    # But wait - in the tensor decomposition, each rank-1 term is U_i * V_i * W_i^T
    # So U_i is a vector, V_i is a vector, W_i is a vector.
    # The product U_i[a] * V_i[b] * W_i[c] gives the coefficient of A[a]*B[b]*C[c].
    
    # For M1 = (x10+x11)*(y00+y01) contributing to Z:
    # We need U to have non-zeros at the positions corresponding to x10 and x11
    # V to have non-zeros at y00 and y01
    # W to have non-zeros at the Z positions this M contributes to
    
    # This means ONE rank-1 term with 2 non-zeros in U, 2 in V, and several in W.
    
    # Let me redefine: add a term with multiple non-zeros in each vector.
    
    U_list = []
    V_list = []
    W_list = []

    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (a_idx, coeff)
        v_entries: list of (b_idx, coeff)  
        w_entries: list of (c_idx, coeff)
        """
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        for idx, coeff in u_entries:
            U_row[idx] = coeff
        for idx, coeff in v_entries:
            V_row[idx] = coeff
        for idx, coeff in w_entries:
            W_row[idx] = coeff
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def aidx(r, c):
        return r * m + c

    def bidx(r, c):
        return r * p + c

    def cidx(r, c):
        return r * n + c

    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for C[c_rows, c_cols] = A[a_rows, a_cols] * B[b_rows, b_cols]
        where all are 2x2 blocks.
        a_rows, b_rows, c_rows: lists of 2 indices
        a_cols, b_cols, c_cols: lists of 2 indices
        """
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        # X = [[x00,x01],[x10,x11]] = A[[ar0,ar1],[ac0,ac1]]
        # Y = [[y00,y01],[y10,y11]] = B[[br0,br1],[bc0,bc1]]
        # Z = [[z00,z01],[z10,z11]] = C[[cr0,cr1],[cc0,cc1]]
        
        # M1 = (x10+x11)(y00+y01)
        # x10 = A[ar1,ac0], x11 = A[ar1,ac1]
        # y00 = B[br0,bc0], y01 = B[br0,bc1]
        # Contributes to: z00 (via -M2+M4+M5+M7... let me use standard)
        
        # Standard Strassen:
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)(y00+y01)
        # M3 = (x00-x10)(y10+y11)
        # M4 = (x01-x11)(y00+y10)
        # M5 = (x00+x01)(y11-y01)
        # M6 = (x10+x11)(y01-y11)
        # M7 = (x01-x10)(y10+y11)
        # z00 = M1 + M4 - M5 + M7
        # z01 = M2 + M3 - M3... 
        
        # Let me use the most common version:
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)(y00+y01)
        # M3 = (x00-x10)(y10+y11)
        # M4 = (x01-x11)(y00+y10)
        # M5 = (x00+x01)(y11-y01)
        # M6 = (x10+x11)(y01-y11)
        # M7 = (x01-x10)(y10+y11)
        # z00 = M1 + M4 - M5 + M7
        # z01 = M2 + M3
        # z10 = M6 + M7
        # z11 = M1 - M2 + M3 + M5
        
        # M1: u=[(ar0,ac0),1],[(ar1,ac1),1], v=[(br0,bc0),1],[(br1,bc1),1]
        # contributes to z00 (+1), z11 (+1)
        add_term(
            [(aidx(ar0,ac0),1), (aidx(ar1,ac1),1)],
            [(bidx(br0,bc0),1), (bidx(br1,bc1),1)],
            [(cidx(cr0,cc0),1), (cidx(cr1,cc1),1)]
        )
        
        # M2: u=[(ar1,ac0),1],[(ar1,ac1),1], v=[(br0,bc0),1],[(br0,bc1),1]
        # contributes to z01 (+1), z11 (-1)
        add_term(
            [(aidx(ar1,ac0),1), (aidx(ar1,ac1),1)],
            [(bidx(br0,bc0),1), (bidx(br0,bc1),1)],
            [(cidx(cr0,cc1),1), (cidx(cr1,cc1),-1)]
        )
        
        # M3: u=[(ar0,ac0),1],[(ar1,ac0),-1], v=[(br1,bc0),1],[(br1,bc1),1]
        # contributes to z01 (+1), z11 (+1)
        add_term(
            [(aidx(ar0,ac0),1), (aidx(ar1,ac0),-1)],
            [(bidx(br1,bc0),1), (bidx(br1,bc1),1)],
            [(cidx(cr0,cc1),1), (cidx(cr1,cc1),1)]
        )
        
        # M4: u=[(ar0,ac1),1],[(ar1,ac1),-1], v=[(br0,bc0),1],[(br1,bc0),1]
        # contributes to z00 (+1)
        add_term(
            [(aidx(ar0,ac1),1), (aidx(ar1,ac1),-1)],
            [(bidx(br0,bc0),1), (bidx(br1,bc0),1)],
            [(cidx(cr0,cc0),1)]
        )
        
        # M5: u=[(ar0,ac0),1],[(ar0,ac1),1], v=[(br1,bc1),1],[(br0,bc1),-1]
        # contributes to z00 (-1), z11 (+1)
        add_term(
            [(aidx(ar0,ac0),1), (aidx(ar0,ac1),1)],
            [(bidx(br1,bc1),1), (bidx(br0,bc1),-1)],
            [(cidx(cr0,cc0),-1), (cidx(cr1,cc1),1)]
        )
        
        # M6: u=[(ar1,ac0),1],[(ar1,ac1),1], v=[(br0,bc1),1],[(br1,bc1),-1]
        # contributes to z10 (+1)
        add_term(
            [(aidx(ar1,ac0),1), (aidx(ar1,ac1),1)],
            [(bidx(br0,bc1),1), (bidx(br1,bc1),-1)],
            [(cidx(cr1,cc0),1)]
        )
        
        # M7: u=[(ar0,ac1),1],[(ar1,ac0),-1], v=[(br1,bc0),1],[(br1,bc1),1]
        # contributes to z00 (+1), z10 (+1)
        add_term(
            [(aidx(ar0,ac1),1), (aidx(ar1,ac0),-1)],
            [(bidx(br1,bc0),1), (bidx(br1,bc1),1)],
            [(cidx(cr0,cc0),1), (cidx(cr1,cc0),1)]
        )

    def schoolbook_2x2x3(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        C[c_rows, c_cols] = A[a_rows, a_cols] * B[b_rows, b_cols]
        A is 2x2, B is 2x3, C is 2x3.
        z00 = x00*y00 + x01*y10
        z01 = x00*y01 + x01*y11
        z02 = x00*y02 + x01*y12
        z10 = x10*y00 + x11*y10
        z11 = x10*y01 + x11*y11
        z12 = x10*y02 + x11*y12
        12 multiplications.
        """
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1, bc2 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1, cc2 = c_cols
        
        # z00
        add_term([(aidx(ar0,ac0),1)], [(bidx(br0,bc0),1)], [(cidx(cr0,cc0),1)])
        add_term([(aidx(ar0,ac1),1)], [(bidx(br1,bc0),1)], [(cidx(cr0,cc0),1)])
        # z01
        add_term([(aidx(ar0,ac0),1)], [(bidx(br0,bc1),1)], [(cidx(cr0,cc1),1)])
        add_term([(aidx(ar0,ac1),1)], [(bidx(br1,bc1),1)], [(cidx(cr0,cc1),1)])
        # z02
        add_term([(aidx(ar0,ac0),1)], [(bidx(br0,bc2),1)], [(cidx(cr0,cc2),1)])
        add_term([(aidx(ar0,ac1),1)], [(bidx(br1,bc2),1)], [(cidx(cr0,cc2),1)])
        # z10
        add_term([(aidx(ar1,ac0),1)], [(bidx(br0,bc0),1)], [(cidx(cr1,cc0),1)])
        add_term([(aidx(ar1,ac1),1)], [(bidx(br1,bc0),1)], [(cidx(cr1,cc0),1)])
        # z11
        add_term([(aidx(ar1,ac0),1)], [(bidx(br0,bc1),1)], [(cidx(cr1,cc1),1)])
        add_term([(aidx(ar1,ac1),1)], [(bidx(br1,bc1),1)], [(cidx(cr1,cc1),1)])
        # z12
        add_term([(aidx(ar1,ac0),1)], [(bidx(br0,bc2),1)], [(cidx(cr1,cc2),1)])
        add_term([(aidx(ar1,ac1),1)], [(bidx(br1,bc2),1)], [(cidx(cr1,cc2),1)])

    def schoolbook_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """2x2 * 2x2 -> 2x2, schoolbook rank 4."""
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        add_term([(aidx(ar0,ac0),1)], [(bidx(br0,bc0),1)], [(cidx(cr0,cc0),1)])
        add_term([(aidx(ar0,ac1),1)], [(bidx(br1,bc0),1)], [(cidx(cr0,cc0),1)])
        add_term([(aidx(ar0,ac0),1)], [(bidx(br0,bc1),1)], [(cidx(cr0,cc1),1)])
        add_term([(aidx(ar0,ac1),1)], [(bidx(br1,bc1),1)], [(cidx(cr0,cc1),1)])
        add_term([(aidx(ar1,ac0),1)], [(bidx(br0,bc0),1)], [(cidx(cr1,cc0),1)])
        add_term([(aidx(ar1,ac1),1)], [(bidx(br1,bc0),1)], [(cidx(cr1,cc0),1)])
        add_term([(aidx(ar1,ac0),1)], [(bidx(br0,bc1),1)], [(cidx(cr1,cc1),1)])
        add_term([(aidx(ar1,ac1),1)], [(bidx(br1,bc1),1)], [(cidx(cr1,cc1),1)])

    # Now build the full 4x4 * 4x7 decomposition.
    # A is 4x4, split into A_top (rows 0,1) and A_bot (rows 2,3), each 2x4.
    # B is 4x7, split into B_top (rows 0,1) and B_bot (rows 2,3), each 2x7.
    # B columns split into: cols 0:2 (2 cols), cols 2:4 (2 cols), cols 4:7 (3 cols)
    
    # C is 4x7.
    # C_top = A_top * B_top + A_bot * B_bot  (2x7)
    # C_bot = A_top * B_bot + A_bot * B_top  (2x7)
    
    # For C_top[0:2, 0:2] = A_top[:,0:2]*B_top[0:2,0:2] + A_top[:,2:4]*B_top[2:4,0:2]
    #   + A_bot[:,0:2]*B_top[0:2,0:2] + A_bot[:,2:4]*B_top[2:4,0:2]
    # Wait, A_top is 2x4 (rows 0,1 of A; all 4 cols). B_top is 2x7 (rows 0,1 of B; all 7 cols).
    # C_top[i,j] = sum_k A[i,k]*B_top[?,j] ... no.
    
    # Actually: C = A * B where A is 4x4, B is 4x7.
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # 
    # Split k into k in {0,1} and k in {2,3}:
    # C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    #
    # Let A_top = A[0:2, :] (2x4), A_bot = A[2:4, :] (2x4)
    # B_top = B[0:2, :] (2x7), B_bot = B[2:4, :] (2x7)
    #
    # C_top = A_top * B_top^T... no.
    # C[0:2, :] = A_top * B_top where B_top is 2x7 but we need to multiply 2x4 * 4x7.
    # A_top is 2x4, and we need B[0:2, :] which is 2x7. That's not matching dimensions.
    #
    # I was confused. Let me redo.
    # A is 4x4: rows 0-3, cols 0-3.
    # B is 4x7: rows 0-3, cols 0-6.
    # C is 4x7: rows 0-3, cols 0-6.
    #
    # C[i,j] = sum_k A[i,k] B[k,j]
    #
    # Split A's rows into top (0,1) and bot (2,3). Split B's rows into top (0,1) and bot (2,3).
    # C[i,j] = sum_{k=0}^{1} A[i,k]B[k,j] + sum_{k=2}^{3} A[i,k]B[k,j]
    #
    # For C_top (i in {0,1}):
    #   C[i,j] = sum_{k=0}^{1} A[i,k]B[k,j] + sum_{k=2}^{3} A[i,k]B[k,j]
    #   = (A_top_row_i) . (B_top_col_j) + (A_bot_row_i) . (B_bot_col_j)
    #   where A_top_row_i = [A[i,0], A[i,1]], A_bot_row_i = [A[i,2], A[i,3]]
    #   B_top_col_j = [B[0,j], B[1,j]], B_bot_col_j = [B[2,j], B[3,j]]
    #
    # So C_top = A_top_mat * B_top_mat + A_bot_mat * B_bot_mat
    # where A_top_mat is 2x2 (A[0:2, 0:2]), A_bot_mat is 2x2 (A[0:2, 2:4])
    # B_top_mat is 2x7 (B[0:2, :]), B_bot_mat is 2x7 (B[2:4, :])
    #
    # So C_top = A[0:2,0:2] * B[0:2,:] + A[0:2,2:4] * B[2:4,:]
    # This is 2x2 * 2x7 + 2x2 * 2x7 = two products of 2x2 by 2x7.
    #
    # Similarly C_bot = A[2:4,0:2] * B[0:2,:] + A[2:4,2:4] * B[2:4,:]
    
    # For 2x2 * 2x7: split B's 7 columns into 2+2+3.
    # 2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x3 (schoolbook, rank 12)
    # Total per 2x2*2x7: 7+7+12 = 26
    #
    # C_top has two such terms: 26+26 = 52
    # C_bot has two such terms: 26+26 = 52
    # Total: 104. That's worse than schoolbook 112? No, 104 < 112.
    #
    # But can we do better? For 2x2*2x2, Strassen is 7 vs schoolbook 4. So 7 > 4!
    # Strassen is only better for larger matrices. For 2x2x2, schoolbook rank is 4, Strassen is 7.
    # So Strassen is WORSE for 2x2x2!
    #
    # I made an error. Strassen reduces exponent but for 2x2x2, schoolbook is rank 4 and Strassen is rank 7.
    # Strassen is beneficial when you have a 4x4x4 = (2x2x2)^4 type decomposition.
    #
    # For 4x4x4: schoolbook is 64. Using 2x2 block decomposition:
    # C = [[A11,A12],[A21,A22]] * [[B11,B12],[B21,B22]]
    # C11 = A11*B11 + A12*B21 (two 2x2x2, rank 4+4=8 each, total 16)
    # C12 = A11*B12 + A12*B22 (16)
    # C21 = A21*B11 + A22*B21 (16)
    # C22 = A21*B12 + A22*B22 (16)
    # Total 64. Same as schoolbook!
    #
    # Strassen for 4x4x4: treat as 2x2 blocks, use Strassen on the 2x2 block matrix.
    # This gives rank 7 * 4 * 4 = 112? No...
    #
    # Actually, the known result: rank of 4x4x4 is at most 27 (Bini et al.) but it's known to be at least 26.
    # The best known upper bound is 27.
    #
    # For 4x4x7, I need a different approach. Let me just use schoolbook with some optimization.
    #
    # Actually, for 2x2*2x7: schoolbook is 2*2*7 = 28. Can we do better?
    # The rank of 2x2x2 is 4, so 2x2*2x2 part needs at least 4. 2x2*2x3 needs at least... 
    #
    # Let me just implement schoolbook for the full thing and see if I can shave off anything.
    # Actually, let me try a different block structure.
    
    # Fresh start with explicit schoolbook but structured.
    U_list = []
    V_list = []
    W_list = []

    for i in range(4):
        for j in range(4):
            for k in range(7):
                a_idx = i * 4 + j
                b_idx = j * 7 + k
                c_idx = k * 4 + i
                U_row = [0.0] * 16
                V_row = [0.0] * 28
                W_row = [0.0] * 28
                U_row[a_idx] = 1.0
                V_row[b_idx] = 1.0
                W_row[c_idx] = 1.0
                U_list.append(U_row)
                V_list.append(V_row)
                W_list.append(W_row)

    rank = len(U_list)
    U = U_list
    V = V_list
    W = W_list
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
