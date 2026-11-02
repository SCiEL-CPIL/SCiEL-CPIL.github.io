def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: C_right = A * B_right (4x4 * 4x1 -> 4x1)
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    # Part 2: C_left = A * B_left (4x4 * 4x4 -> 4x4)
    # Strassen on 2x2 blocks.
    # A blocks: A00 (0,0;0,1;1,0;1,1), A01, A10, A11
    # B blocks: B00, B01, B10, B11 (cols 0-3 of B)
    
    # Define Strassen factors for 2x2x2
    # Inputs X, Y (2x2). Output Z.
    # X indices: x00, x01, x10, x11
    # Y indices: y00, y01, y10, y11
    
    # We need to express X and Y for each of the 7 block multiplications M1..M7
    # in terms of A and B_left entries.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Let's define the linear forms for X and Y for each Mk.
    # X is 2x2, Y is 2x2.
    # We will iterate k from 0 to 6.
    
    # Helper to get A entry index: a(i, j) -> i*4+j
    # Helper to get B entry index: b(i, j) -> i*5+j (j is 0..3 for left part)
    
    # For each Mk, we define X (2x2) and Y (2x2) as linear combos of A and B.
    # Then we apply Strassen to X*Y.
    
    # Strassen scalar multiplications for X*Y:
    # s1 = (x00+x11)(y00+y11)
    # s2 = (x10+x11)y00
    # s3 = x00(y01-y11)
    # s4 = x11(y10-y00)
    # s5 = (x00+x01)y11
    # s6 = (x10-x00)(y00+y01)
    # s7 = (x01-x11)(y10+y11)
    
    # We need to expand each term into U, V, W contributions.
    # Each scalar multiplication is of the form (linear_A) * (linear_B).
    # So for each scalar mult, we add 1 to U for each A term in linear_A with its coeff,
    # and 1 to V for each B term in linear_B with its coeff.
    # And W gets the coefficient for the corresponding C entry.
    
    # Let's define the linear forms for X and Y for each k.
    # x00, x01, x10, x11 are linear forms in A.
    # y00, y01, y10, y11 are linear forms in B.
    
    # We represent a linear form as a dict {index: coeff}
    
    def add_linear_form(U_mat, V_mat, W_mat, col_idx, form_U, form_V, w_coeffs):
        # form_U: dict {A_idx: coeff}
        # form_V: dict {B_idx: coeff}
        # w_coeffs: dict {C_idx: coeff}
        # This adds one rank-1 term? No, this is one scalar multiplication.
        # The scalar multiplication is sum_U * sum_V.
        # So we add to U[i][col] the coeff for each i in form_U.
        # And to V[j][col] the coeff for each j in form_V.
        # And to W[k][col] the coeff for each k in w_coeffs.
        # Wait, the rank-1 term is u * v * w.
        # u is a vector in R^(nm), v in R^(mp), w in R^(pn).
        # If the scalar multiplication is (sum a_i u_i) * (sum b_j v_j),
        # then the term is sum a_i b_j u_i v_j.
        # This is not a single rank-1 term in the tensor decomposition unless a_i and b_j are singletons.
        # Strassen's algorithm uses scalar multiplications of linear combinations.
        # The tensor decomposition requires rank-1 terms of the form u e_i v e_j w e_k.
        # A multiplication of linear forms (sum a_i u_i) (sum b_j v_j) expands to sum a_i b_j u_i v_j.
        # Each u_i v_j is a rank-1 term (with appropriate w).
        # So one Strassen "multiplication" step actually adds multiple rank-1 terms to the decomposition?
        # No. The rank of the tensor is the number of rank-1 terms in the decomposition.
        # Strassen reduces the number of *scalar multiplications* in the algorithm.
        # However, the rank of the tensor is an algebraic invariant.
        # The number of scalar multiplications in an algorithm is an upper bound on the rank?
        # No, the rank is the minimum number of rank-1 tensors.
        # Strassen's algorithm for 2x2x2 uses 7 scalar multiplications.
        # The rank of the 2x2x2 tensor is 7.
        # So the 7 scalar multiplications correspond to the 7 rank-1 terms.
        # BUT, in Strassen, the inputs to the scalar multiplications are linear combinations.
        # The rank-1 term is $u \otimes v \otimes w$.
        # $u$ and $v$ are vectors.
        # If the scalar multiplication is $L_A \cdot L_B$, where $L_A = \sum a_i A_i$ and $L_B = \sum b_j B_j$,
        # then the contribution to the output is $L_A L_B W_{out}$.
        # This is $\sum_{i,j} a_i b_j A_i B_j W_{out}$.
        # This is a sum of rank-1 terms $A_i \otimes B_j \otimes W_{out}$ with coefficients $a_i b_j$.
        # So ONE scalar multiplication in Strassen corresponds to MULTIPLE rank-1 terms in the tensor decomposition?
        # No! The rank-1 term in the tensor decomposition is defined by the vectors U, V, W.
        # If we have a term $U_r \cdot V_r \cdot W_r$, it contributes $U_r[i] V_r[j] W_r[k]$ to $C[i,k]$.
        # In Strassen, we compute $M = L_A L_B$.
        # Then we add $M$ to some $C$ entries.
        # So $C[k] += M \cdot c_k$.
        # $C[k] += (\sum_i a_i A_i) (\sum_j b_j B_j) c_k = \sum_{i,j} a_i b_j c_k A_i B_j$.
        # This looks like we are adding many rank-1 terms.
        # However, we can absorb the coefficients into U and V.
        # Let $U_r[i] = a_i$ and $V_r[j] = b_j$.
        # Then the term is $\sum_{i,j} U_r[i] V_r[j] (c_k W_r[k])$.
        # This is exactly one rank-1 term with vectors U_r, V_r, W_r.
        # So YES, one Strassen scalar multiplication corresponds to ONE rank-1 term in the decomposition.
        # The vectors U_r and V_r contain the coefficients of the linear forms.
        
        for idx, coeff in form_U.items():
            U_mat[idx][col_idx] += coeff
        for idx, coeff in form_V.items():
            V_mat[idx][col_idx] += coeff
        for idx, coeff in w_coeffs.items():
            W_mat[idx][col_idx] += coeff

    # Define the linear forms for each M_k
    # A indices: 0..15. A[i*4+j]
    # B indices: 0..19. B[i*5+j] (j=0..3 for left)
    
    # M1: X = A00 + A11, Y = B00 + B11
    # A00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    # B00: (0,0)..(1,1) -> 0,1,5,6
    # B11: (2,2)..(3,3) -> 12,13,17,18
    
    # We need to define X (2x2) and Y (2x2) for each M_k.
    # X entries are linear forms in A.
    # Y entries are linear forms in B.
    
    # Let's create a function to get the dict for a block.
    def get_A_block(r, c):
        # r, c in 0..1. Block size 2x2.
        # Rows 2r, 2r+1. Cols 2c, 2c+1.
        d = {}
        for i in range(2):
            for j in range(2):
                idx = (2*r + i)*4 + (2*c + j)
                d[idx] = 1
        return d

    def get_B_block(r, c):
        d = {}
        for i in range(2):
            for j in range(2):
                idx = (2*r + i)*5 + (2*c + j)
                d[idx] = 1
        return d

    def add_dict(d1, d2, sign=1):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) + sign * v
        return d

    def sub_dict(d1, d2):
        return add_dict(d1, d2, -1)

    # M1: X = A00 + A11, Y = B00 + B11
    # X is 2x2? No, X and Y are 2x2 matrices in the Strassen recursion.
    # Wait. The block level Strassen operates on 2x2 blocks of size 2x2.
    # So X and Y are 2x2 matrices.
    # But in M1, X = A00 + A11. This is a 2x2 matrix.
    # So X00 = A0000 + A1100, etc.
    # We need to define the 4 entries of X and 4 entries of Y.
    
    # Let's define X and Y for each k as dicts of dicts?
    # Or just 4 dicts for X (x00, x01, x10, x11) and 4 for Y.
    
    # M1:
    # X = A00 + A11. So x00 = A0000+A1100, x01 = A0001+A1101, x10 = A0010+A1110, x11 = A0011+A1111.
    # Y = B00 + B11.
    
    # Let's write a helper to get the linear form for X[i][j] and Y[i][j].
    
    # For M1:
    X1 = [
        add_dict(get_A_block(0,0)[0:2] if False else {0:1, 1:1, 4:1, 5:1} # A00
             , {10:1, 11:1, 14:1, 15:1}), # A11
        # ... this is getting messy.
    ]
    
    # Let's simplify.
    # We will define the 4 linear forms for X and 4 for Y for each M_k.
    
    def get_A_lin(r, c, i, j):
        # Linear form for A_block[r][c] entry [i][j]
        # This is just A[2r+i][2c+j]
        idx = (2*r+i)*4 + (2*c+j)
        return {idx: 1}

    def get_B_lin(r, c, i, j):
        idx = (2*r+i)*5 + (2*c+j)
        return {idx: 1}

    # M1: X = A00 + A11, Y = B00 + B11
    # X00 = A0000 + A1100.
    # X01 = A0001 + A1101.
    # X10 = A0010 + A1110.
    # X11 = A0011 + A1111.
    
    # We can just sum the dicts.
    
    def solve():
        pass

    # Let's implement the 7 M_k definitions.
    
    # M1
    X1 = [
        add_dict(get_A_lin(0,0,0,0), get_A_lin(1,1,0,0)),
        add_dict(get_A_lin(0,0,0,1), get_A_lin(1,1,0,1)),
        add_dict(get_A_lin(0,0,1,0), get_A_lin(1,1,1,0)),
        add_dict(get_A_lin(0,0,1,1), get_A_lin(1,1,1,1))
    ]
    Y1 = [
        add_dict(get_B_lin(0,0,0,0), get_B_lin(1,1,0,0)),
        add_dict(get_B_lin(0,0,0,1), get_B_lin(1,1,0,1)),
        add_dict(get_B_lin(0,0,1,0), get_B_lin(1,1,1,0)),
        add_dict(get_B_lin(0,0,1,1), get_B_lin(1,1,1,1))
    ]
    
    # M2: X = A10 + A11, Y = B00
    X2 = [
        add_dict(get_A_lin(1,0,0,0), get_A_lin(1,1,0,0)),
        add_dict(get_A_lin(1,0,0,1), get_A_lin(1,1,0,1)),
        add_dict(get_A_lin(1,0,1,0), get_A_lin(1,1,1,0)),
        add_dict(get_A_lin(1,0,1,1), get_A_lin(1,1,1,1))
    ]
    Y2 = [
        get_B_lin(0,0,0,0), get_B_lin(0,0,0,1),
        get_B_lin(0,0,1,0), get_B_lin(0,0,1,1)
    ]
    
    # M3: X = A00, Y = B01 - B11
    X3 = [
        get_A_lin(0,0,0,0), get_A_lin(0,0,0,1),
        get_A_lin(0,0,1,0), get_A_lin(0,0,1,1)
    ]
    Y3 = [
        sub_dict(get_B_lin(0,1,0,0), get_B_lin(1,1,0,0)),
        sub_dict(get_B_lin(0,1,0,1), get_B_lin(1,1,0,1)),
        sub_dict(get_B_lin(0,1,1,0), get_B_lin(1,1,1,0)),
        sub_dict(get_B_lin(0,1,1,1), get_B_lin(1,1,1,1))
    ]
    
    # M4: X = A11, Y = B10 - B00
    X4 = [
        get_A_lin(1,1,0,0), get_A_lin(1,1,0,1),
        get_A_lin(1,1,1,0), get_A_lin(1,1,1,1)
    ]
    Y4 = [
        sub_dict(get_B_lin(1,0,0,0), get_B_lin(0,0,0,0)),
        sub_dict(get_B_lin(1,0,0,1), get_B_lin(0,0,0,1)),
        sub_dict(get_B_lin(1,0,1,0), get_B_lin(0,0,1,0)),
        sub_dict(get_B_lin(1,0,1,1), get_B_lin(0,0,1,1))
    ]
    
    # M5: X = A00 + A01, Y = B11
    X5 = [
        add_dict(get_A_lin(0,0,0,0), get_A_lin(0,1,0,0)),
        add_dict(get_A_lin(0,0,0,1), get_A_lin(0,1,0,1)),
        add_dict(get_A_lin(0,0,1,0), get_A_lin(0,1,1,0)),
        add_dict(get_A_lin(0,0,1,1), get_A_lin(0,1,1,1))
    ]
    Y5 = [
        get_B_lin(1,1,0,0), get_B_lin(1,1,0,1),
        get_B_lin(1,1,1,0), get_B_lin(1,1,1,1)
    ]
    
    # M6: X = A10 - A00, Y = B00 + B01
    X6 = [
        sub_dict(get_A_lin(1,0,0,0), get_A_lin(0,0,0,0)),
        sub_dict(get_A_lin(1,0,0,1), get_A_lin(0,0,0,1)),
        sub_dict(get_A_lin(1,0,1,0), get_A_lin(0,0,1,0)),
        sub_dict(get_A_lin(1,0,1,1), get_A_lin(0,0,1,1))
    ]
    Y6 = [
        add_dict(get_B_lin(0,0,0,0), get_B_lin(0,1,0,0)),
        add_dict(get_B_lin(0,0,0,1), get_B_lin(0,1,0,1)),
        add_dict(get_B_lin(0,0,1,0), get_B_lin(0,1,1,0)),
        add_dict(get_B_lin(0,0,1,1), get_B_lin(0,1,1,1))
    ]
    
    # M7: X = A01 - A11, Y = B10 + B11
    X7 = [
        sub_dict(get_A_lin(0,1,0,0), get_A_lin(1,1,0,0)),
        sub_dict(get_A_lin(0,1,0,1), get_A_lin(1,1,0,1)),
        sub_dict(get_A_lin(0,1,1,0), get_A_lin(1,1,1,0)),
        sub_dict(get_A_lin(0,1,1,1), get_A_lin(1,1,1,1))
    ]
    Y7 = [
        add_dict(get_B_lin(1,0,0,0), get_B_lin(1,1,0,0)),
        add_dict(get_B_lin(1,0,0,1), get_B_lin(1,1,0,1)),
        add_dict(get_B_lin(1,0,1,0), get_B_lin(1,1,1,0)),
        add_dict(get_B_lin(1,0,1,1), get_B_lin(1,1,1,1))
    ]
    
    # Now, for each M_k, we have X (4 linear forms) and Y (4 linear forms).
    # We apply Strassen to X*Y.
    # This generates 7 scalar multiplications.
    # For each scalar multiplication, we determine the W coefficients.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C is 4x4.
    # C00 is block (0,0). Entries (0,0),(0,1),(1,0),(1,1).
    # C01 is block (0,1). Entries (0,2),(0,3),(1,2),(1,3).
    # C10 is block (1,0). Entries (2,0),(2,1),(3,0),(3,1).
    # C11 is block (1,1). Entries (2,2),(2,3),(3,2),(3,3).
    
    # W indices for C_left:
    # C[i,j] -> W[j*4 + i] ? No, W[k*n + i].
    # Here n=4, p=5.
    # For C_left, k is 0..3.
    # W index for C[i,k] is k*4 + i.
    
    # Let's define the W coefficients for each of the 7 Strassen steps for a given M_k.
    # The Strassen steps produce z00, z01, z10, z11 of the 2x2 product X*Y.
    # These z's are the entries of M_k.
    # M_k contributes to C blocks.
    
    # For M1:
    # M1 contributes to C00 (+1), C11 (+1).
    # So z00 of M1 adds to C0000, C0001, C0010, C0011?
    # No, M1 is a 2x2 matrix.
    # M1[0,0] adds to C[0,0]. M1[0,1] adds to C[0,1].
    # M1[1,0] adds to C[1,0]. M1[1,1] adds to C[1,1].
    
    # So for M1:
    # s1 (z00) -> C0000
    # s2 (z10) -> C0100? No.
    # Strassen z indices:
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # So s1 contributes to z00 (+1) and z11 (+1).
    # s2 contributes to z10 (+1) and z11 (-1).
    # s3 contributes to z01 (+1) and z11 (+1).
    # s4 contributes to z00 (+1) and z10 (+1).
    # s5 contributes to z00 (-1) and z01 (+1).
    # s6 contributes to z11 (+1).
    # s7 contributes to z00 (+1).
    
    # We need to map these to W indices.
    # For M1:
    # z00 -> C0000 (idx 0*4+0=0)
    # z01 -> C0001 (idx 0*4+1=1)? No, C[0,1] is row 0, col 1. W index 1*4+0=4?
    # W index is k*n + i. k is column, i is row.
    # C[0,0] -> k=0, i=0 -> 0.
    # C[0,1] -> k=1, i=0 -> 4.
    # C[1,0] -> k=0, i=1 -> 1.
    # C[1,1] -> k=1, i=1 -> 5.
    
    # So for M1:
    # s1: W[0] += 1, W[2*4+2] += 1 (C1100? No, C11 is block 1,1. C[2,2] -> k=2, i=2 -> 10).
    # Wait, C11 block is rows 2,3 cols 2,3.
    # M1[1,1] is C[2,2]? No.
    # M1 is the product of block 00 and 11? No.
    # M1 = (A00+A11)(B00+B11).
    # This product is a 2x2 matrix.
    # Let's call it P1.
    # P1[0,0] adds to C00[0,0] = C[0,0].
    # P1[0,1] adds to C00[0,1] = C[0,1].
    # P1[1,0] adds to C00[1,0] = C[1,0].
    # P1[1,1] adds to C00[1,1] = C[1,1].
    # AND
    # P1[0,0] adds to C11[0,0] = C[2,2].
    # P1[0,1] adds to C11[0,1] = C[2,3].
    # P1[1,0] adds to C11[1,0] = C[3,2].
    # P1[1,1] adds to C11[1,1] = C[3,3].
    
    # So s1 (z00) contributes to P1[0,0] and P1[1,1].
    # So s1 adds to C[0,0], C[1,1], C[2,2], C[3,3].
    # W indices: 0, 5, 10, 15.
    
    # This is getting complex but doable.
    
    # Let's write a helper to add the W coefficients for a given s_j in M_k.
    
    def get_w_coeffs(M_k, s_j):
        # Returns dict {W_idx: coeff}
        coeffs = {}
        # Determine z_idx and coeff from s_j
        # s1: z00(+1), z11(+1)
        # s2: z10(+1), z11(-1)
        # s3: z01(+1), z11(+1)
        # s4: z00(+1), z10(+1)
        # s5: z00(-1), z01(+1)
        # s6: z11(+1)
        # s7: z00(+1)
        
        z_contribs = {
            1: [(0, 1), (3, 1)], # s1: z00, z11
            2: [(2, 1), (3, -1)], # s2: z10, z11
            3: [(1, 1), (3, 1)], # s3: z01, z11
            4: [(0, 1), (2, 1)], # s4: z00, z10
            5: [(0, -1), (1, 1)], # s5: z00, z01
            6: [(3, 1)], # s6: z11
            7: [(0, 1)] # s7: z00
        }
        
        for z_idx, z_coeff in z_contribs[s_j]:
            # Map z_idx to C block entry
            # z00 -> C_block[0,0]
            # z01 -> C_block[0,1]
            # z10 -> C_block[1,0]
            # z11 -> C_block[1,1]
            
            # Determine which C blocks M_k contributes to and with what sign.
            # M1: C00(+), C11(+)
            # M2: C10(+)
            # M3: C01(+), C11(+)
            # M4: C00(+), C10(+)
            # M5: C00(-), C01(+)
            # M6: C11(+)
            # M7: C00(+)
            
            # Define contribution map for each M_k
            # Map M_k to list of (C_block_row, C_block_col, sign)
            contrib_map = {
                1: [(0,0,1), (1,1,1)],
                2: [(1,0,1)],
                3: [(0,1,1), (1,1,1)],
                4: [(0,0,1), (1,0,1)],
                5: [(0,0,-1), (0,1,1)],
                6: [(1,1,1)],
                7: [(0,0,1)]
            }
            
            for br, bc, sign in contrib_map[M_k]:
                # C block (br, bc) entry (z_row, z_col)
                # z_row, z_col from z_idx
                zr = z_idx // 2
                zc = z_idx % 2
                # Global row = 2*br + zr
                # Global col = 2*bc + zc
                gr = 2*br + zr
                gc = 2*bc + zc
                w_idx = gc * 4 + gr
                coeffs[w_idx] = coeffs.get(w_idx, 0) + z_coeff * sign
                
        return coeffs

    # Now loop through M1..M7 and s1..s7
    for k in range(1, 8):
        X = [X1, X2, X3, X4, X5, X6, X7][k-1]
        Y = [Y1, Y2, Y3, Y4, Y5, Y6, Y7][k-1]
        for s in range(1, 8):
            # Determine U and V linear forms for this s
            # s1: (x00+x11)(y00+y11)
            # s2: (x10+x11)y00
            # s3: x00(y01-y11)
            # s4: x11(y10-y00)
            # s5: (x00+x01)y11
            # s6: (x10-x00)(y00+y01)
            # s7: (x01-x11)(y10+y11)
            
            if s == 1:
                u_form = add_dict(X[0], X[3])
                v_form = add_dict(Y[0], Y[3])
            elif s == 2:
                u_form = add_dict(X[2], X[3])
                v_form = Y[0]
            elif s == 3:
                u_form = X[0]
                v_form = sub_dict(Y[1], Y[3])
            elif s == 4:
                u_form = X[3]
                v_form = sub_dict(Y[2], Y[0])
            elif s == 5:
                u_form = add_dict(X[0], X[1])
                v_form = Y[3]
            elif s == 6:
                u_form = sub_dict(X[2], X[0])
                v_form = add_dict(Y[0], Y[1])
            elif s == 7:
                u_form = sub_dict(X[1], X[3])
                v_form = add_dict(Y[2], Y[3])
            else:
                continue
                
            w_coeffs = get_w_coeffs(k, s)
            
            add_linear_form(U, V, W, col, u_form, v_form, w_coeffs)
            col += 1
            
    return {"U": U, "V": V, "W": W}
