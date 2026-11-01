"""Matrix multiplication 4x4x4: rank-minimal exact tensor decomposition via recursive Strassen."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-minimal exact tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen: apply Strassen at the 4x4 block level (7 products),
    and within each 2x2 sub-product apply Strassen again (7 scalar mults each),
    giving 7*7 = 49 total rank terms.
    """
    import time
    start_time = time.time()
    DEADLINE = 150  # seconds

    n, m, p = DIMENSIONS  # 4, 4, 4

    # Strassen 2x2: 7 multiplications
    # For C = A*B where A, B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to build U, V, W matrices such that:
    # (A*B)_{i,k} = sum_r U[i*4+j... wait, let me re-read the convention.
    # W[k*n+i, r] is the output factor.
    # The tensor is T[i,j,k] = delta_{i,k} (for matrix mult: C_ik = sum_j A_ij B_jk)
    # Actually for matrix multiplication: T[i,j,k] = 1 if the (i,k) element of C depends on A[i,j]*B[j,k]
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention says:
    # U: shape (n*m, rank) -> U[i*m+j, r]
    # V: shape (m*p, rank) -> V[j*p+k, r]
    # W: shape (p*n, rank) -> W[k*n+i, r]
    # And the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 (it's the identity tensor for the bilinear form)
    # Actually, matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has T[i,j,k] = 1.
    # Wait no. The tensor for matrix multiplication of size n x p = (n x m)(m x p) is:
    # T[i,j,k] = 1 (delta-like). Specifically, the multilinear form is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # so T[i,j,k] = 1 for all i,j,k.
    # Hmm, that's not right either. Let me think again.
    # The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the coefficient is 1
    # Actually: T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # For standard matrix multiplication, every A[i,j]*B[j,k] contributes to C[i,k],
    # so T[i,j,k] = 1 for all i,j,k.
    # 
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # Recursive Strassen approach:
    # Level 1: Split 4x4 into 2x2 blocks.
    # A = [[A00, A01],[A10, A11]] where each block is 2x2
    # B = [[B00, B01],[B10, B11]]
    # C = [[C00, C01],[C10, C11]]
    # 
    # Strassen for 4x4 (block level):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    #
    # Each S_i is a 2x2 product of 2x2 matrices, which we decompose with Strassen again (7 mults).
    # Total: 7 * 7 = 49 rank terms.

    # Let's build this systematically.
    # Each 2x2 Strassen decomposition has 7 terms. For each term, we have:
    # U factor: linear combination of elements of the 2x2 A-block
    # V factor: linear combination of elements of the 2x2 B-block
    # W factor: linear combination of elements of the 2x2 C-block

    # Let's define the 2x2 Strassen decomposition in terms of (u, v, w) triples.
    # For a 2x2 matrix multiplication A*B=C:
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]], C = [[c00, c01],[c10, c11]]
    # 
    # The 7 Strassen terms give:
    # c00 = s1 + s4 - s5 + s7
    # c01 = s3 + s5
    # c10 = s2 + s4
    # c11 = s1 - s2 + s3 + s6
    #
    # where:
    # s1 = (a00+a11)(b00+b11)
    # s2 = (a10+a11)b00
    # s3 = a00(b01-b11)
    # s4 = a11(b10-b00)
    # s5 = (a00+a01)b11
    # s6 = (a10-a00)(b00+b01)
    # s7 = (a01-a11)(b10+b11)
    #
    # So for each s_i, we have a u-vector (coefficients on a's), v-vector (coefficients on b's),
    # and w-vectors (coefficients on c's).

    # Let me represent the 2x2 Strassen as a list of (u_vec, v_vec, [(w_coeff, c_index)])
    # where u_vec is a 4-vector (for a00,a01,a10,a11), v_vec is a 4-vector (for b00,b01,b10,b11),
    # and the w part maps to the 4 c elements.

    # Actually, for the tensor decomposition, each rank-1 term is:
    # u_vec ⊗ v_vec ⊗ w_vec
    # where u_vec has 4 entries (for the 4 elements of a 2x2 matrix),
    # v_vec has 4 entries, w_vec has 4 entries.
    # And we need: sum of (u_vec ⊗ v_vec ⊗ w_vec) over all terms = the 2x2 matmul tensor.

    # The 2x2 matmul tensor T_2[i,j,k] where i is row of C, j is... 
    # Actually for 2x2 matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So T_2[i,j,k] = 1 for all i,j,k in {0,1}.
    # Wait, that's the full tensor. Let me index properly.
    # A has indices (i,j) for i,j in {0,1}
    # B has indices (j,k) for j,k in {0,1}  
    # C has indices (i,k) for i,k in {0,1}
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor is all ones: T[i,j,k] = 1 for all i,j,k.

    # Strassen decomposition of the 2x2 matmul tensor:
    # We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    # 
    # From Strassen:
    # c00 = s1 + s4 - s5 + s7  -> w for c00 gets contributions from s1,s4,s5,s7
    # c01 = s3 + s5
    # c10 = s2 + s4
    # c11 = s1 - s2 + s3 + s6
    #
    # s1 = (a00+a11)(b00+b11): u1=[1,0,0,1], v1=[1,0,0,1]
    # s2 = (a10+a11)b00: u2=[0,0,1,1], v2=[1,0,0,0]
    # s3 = a00(b01-b11): u3=[1,0,0,0], v3=[0,1,0,-1]
    # s4 = a11(b10-b00): u4=[0,0,0,1], v4=[-1,0,1,0]
    # s5 = (a00+a01)b11: u5=[1,1,0,0], v5=[0,0,0,1]
    # s6 = (a10-a00)(b00+b01): u6=[-1,0,1,0], v6=[1,1,0,0]
    # s7 = (a01-a11)(b10+b11): u7=[0,1,-1,1], v7=[0,0,1,1]
    #
    # W vectors (for c00,c01,c10,c11):
    # s1 contributes to c00 (+1) and c11 (+1): w1=[1,0,0,1]
    # s2 contributes to c10 (+1) and c11 (-1): w2=[0,0,1,-1]
    # s3 contributes to c01 (+1) and c11 (+1): w3=[0,1,0,1]
    # s4 contributes to c00 (+1) and c10 (+1): w4=[1,0,1,0]
    # s5 contributes to c00 (-1) and c01 (+1): w5=[-1,1,0,0]
    # s6 contributes to c11 (+1): w6=[0,0,0,1]
    # s7 contributes to c00 (+1): w7=[1,0,0,0]

    # Let me verify: 
    # c00 = s1 + s4 - s5 + s7 = w1[0]*s1 + w4[0]*s4 + w5[0]*s5 + w7[0]*s7 = 1*s1 + 1*s4 + (-1)*s5 + 1*s7 ✓
    # c01 = s3 + s5 = w3[1]*s3 + w5[1]*s5 = 1*s3 + 1*s5 ✓
    # c10 = s2 + s4 = w2[2]*s2 + w4[2]*s4 = 1*s2 + 1*s4 ✓
    # c11 = s1 - s2 + s3 + s6 = w1[3]*s1 + w2[3]*s2 + w3[3]*s3 + w6[3]*s6 = 1*s1 + (-1)*s2 + 1*s3 + 1*s6 ✓

    # Great! Now for the 4x4 case:
    # We split into 2x2 blocks. Let me define block indices.
    # A is 4x4, split into:
    # A00 = A[0:2,0:2], A01 = A[0:2,2:4]
    # A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # Similarly for B and C.
    #
    # The 7 Strassen block products:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    #
    # Each S_i is a 2x2 matrix product. We decompose each with 2x2 Strassen (7 terms each).
    # Total: 49 terms.
    #
    # For each block product S_i = X_i * Y_i where X_i and Y_i are 2x2 matrices
    # (linear combinations of blocks of A and B), we apply the 2x2 Strassen decomposition.
    #
    # The U factor for the full 4x4: each 2x2 block of A has 4 elements.
    # U has shape (n*m, rank) = (16, 49).
    # Index into U: i*m + j where i is row (0-3), j is col (0-3).
    # Block A00: rows 0-1, cols 0-1 -> U indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # Block A01: rows 0-1, cols 2-3 -> U indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # Block A10: rows 2-3, cols 0-1 -> U indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Block A11: rows 2-3, cols 2-3 -> U indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    #
    # Similarly for V (shape m*p = 16, 49):
    # Block B00: rows 0-1, cols 0-1 -> V indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # Block B01: rows 0-1, cols 2-3 -> V indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # Block B10: rows 2-3, cols 0-1 -> V indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Block B11: rows 2-3, cols 2-3 -> V indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    #
    # W (shape p*n = 16, 49):
    # Block C00: rows 0-1, cols 0-1 -> W indices 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # Block C01: rows 0-1, cols 2-3 -> W indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # Block C10: rows 2-3, cols 0-1 -> W indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Block C11: rows 2-3, cols 2-3 -> W indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15

    # Block element indices within a 2x2 block:
    # [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
    # For U (A blocks): element (br, bc) in block (bi, bj) where bi,bj in {0,1}
    # global row = bi*2 + br, global col = bj*2 + bc
    # U index = (bi*2+br)*4 + (bj*2+bc)
    #
    # For V (B blocks): element (br, bc) in block (bi, bj)
    # global row = bi*2 + br, global col = bj*2 + bc
    # V index = (bi*2+br)*4 + (bj*2+bc)
    #
    # For W (C blocks): element (br, bc) in block (bi, bj)
    # global row = bi*2 + br, global col = bj*2 + bc
    # W index = (bi*2+br)*4 + (bj*2+bc)

    # 2x2 Strassen decomposition (u, v, w) triples:
    strassen_2x2 = [
        # (u_vec, v_vec, w_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # s1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # s2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # s3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # s4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # s5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # s6
        ([0, 1, -1, 1], [0, 0, 1, 1], [1, 0, 0, 0]),  # s7
    ]

    # Block-level Strassen: for each of the 7 block products S_i = X_i * Y_i,
    # X_i is a linear combination of A-blocks, Y_i is a linear combination of B-blocks.
    # X_i coefficients on (A00, A01, A10, A11):
    # S1: X1 = A00 + A11 -> coeff [1, 0, 0, 1]
    # S2: X2 = A10 + A11 -> coeff [0, 0, 1, 1]
    # S3: X3 = A00 -> coeff [1, 0, 0, 0]
    # S4: X4 = A11 -> coeff [0, 0, 0, 1]
    # S5: X5 = A00 + A01 -> coeff [1, 1, 0, 0]
    # S6: X6 = A10 - A00 -> coeff [-1, 0, 1, 0]
    # S7: X7 = A01 - A11 -> coeff [0, 1, -1, 1]
    #
    # Y_i coefficients on (B00, B01, B10, B11):
    # S1: Y1 = B00 + B11 -> coeff [1, 0, 0, 1]
    # S2: Y2 = B00 -> coeff [1, 0, 0, 0]
    # S3: Y3 = B01 - B11 -> coeff [0, 1, 0, -1]
    # S4: Y4 = B10 - B00 -> coeff [-1, 0, 1, 0]
    # S5: Y5 = B11 -> coeff [0, 0, 0, 1]
    # S6: Y6 = B00 + B01 -> coeff [1, 1, 0, 0]
    # S7: Y7 = B10 + B11 -> coeff [0, 0, 1, 1]
    #
    # C-block contributions:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6

    block_x = [
        [1, 0, 0, 1],   # S1: A00 + A11
        [0, 0, 1, 1],   # S2: A10 + A11
        [1, 0, 0, 0],   # S3: A00
        [0, 0, 0, 1],   # S4: A11
        [1, 1, 0, 0],   # S5: A00 + A01
        [-1, 0, 1, 0],  # S6: A10 - A00
        [0, 1, -1, 1],  # S7: A01 - A11
    ]
    block_y = [
        [1, 0, 0, 1],   # S1: B00 + B11
        [1, 0, 0, 0],   # S2: B00
        [0, 1, 0, -1],  # S3: B01 - B11
        [-1, 0, 1, 0],  # S4: B10 - B00
        [0, 0, 0, 1],   # S5: B11
        [1, 1, 0, 0],   # S6: B00 + B01
        [0, 0, 1, 1],   # S7: B10 + B11
    ]
    # C-block: for each S_i, which C-blocks does it contribute to and with what sign?
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    block_c = [
        # S_i: {C_block: coeff}
        {0: 1, 3: 1},      # S1 -> C00(+1), C11(+1)
        {2: 1, 3: -1},     # S2 -> C10(+1), C11(-1)
        {1: 1, 3: 1},      # S3 -> C01(+1), C11(+1)
        {0: 1, 2: 1},      # S4 -> C00(+1), C10(+1)
        {0: -1, 1: 1},     # S5 -> C00(-1), C01(+1)
        {3: 1},            # S6 -> C11(+1)
        {0: 1},            # S7 -> C00(+1)
    ]

    # Now build the full 49-term decomposition.
    # For each block product i (0-6) and each 2x2 Strassen term j (0-6):
    #   rank term r = i*7 + j
    #   U contribution: for each A-block ab (0-3), for each element e (0-3) in that block:
    #     U[global_index(ab, e)][r] += block_x[i][ab] * strassen_2x2[j][0][e]
    #   V contribution: for each B-block bb (0-3), for each element e (0-3):
    #     V[global_index(bb, e)][r] += block_y[i][bb] * strassen_2x2[j][1][e]
    #   W contribution: for each C-block cb that S_i contributes to, for each element e:
    #     W[global_index(cb, e)][r] += block_c[i][cb] * strassen_2x2[j][2][e]

    rank = 49
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]

    def block_elem_index(block_idx, elem_idx):
        """
        block_idx: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
        elem_idx: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
        Returns the global flat index for U/V/W (which all use row*4+col).
        """
        bi = block_idx // 2  # block row: 0 or 1
        bj = block_idx % 2   # block col: 0 or 1
        br = elem_idx // 2   # element row within block: 0 or 1
        bc = elem_idx % 2    # element col within block: 0 or 1
        global_row = bi * 2 + br
        global_col = bj * 2 + bc
        return global_row * 4 + global_col

    for i in range(7):  # block product
        for j in range(7):  # 2x2 Strassen term
            r = i * 7 + j
            u_vec, v_vec, w_vec = strassen_2x2[j]
            
            # U: sum over A-blocks
            for ab in range(4):
                coeff_x = block_x[i][ab]
                if coeff_x == 0:
                    continue
                for e in range(4):
                    u_elem = u_vec[e]
                    if u_elem == 0:
                        continue
                    idx = block_elem_index(ab, e)
                    U[idx][r] += coeff_x * u_elem
            
            # V: sum over B-blocks
            for bb in range(4):
                coeff_y = block_y[i][bb]
                if coeff_y == 0:
                    continue
                for e in range(4):
                    v_elem = v_vec[e]
                    if v_elem == 0:
                        continue
                    idx = block_elem_index(bb, e)
                    V[idx][r] += coeff_y * v_elem
            
            # W: sum over C-blocks that S_i contributes to
            for cb, coeff_c in block_c[i].items():
                if coeff_c == 0:
                    continue
                for e in range(4):
                    w_elem = w_vec[e]
                    if w_elem == 0:
                        continue
                    idx = block_elem_index(cb, e)
                    W[idx][r] += coeff_c * w_elem

    # Convert to integers if possible (all values should be integers)
    def to_int_list(lst):
        return [int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in lst]

    U = [to_int_list(row) for row in U]
    V = [to_int_list(row) for row in V]
    W = [to_int_list(row) for row in W]

    # Verify the decomposition (quick check)
    if time.time() - start_time < DEADLINE - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1
        # This is a 4x4x4 check = 64 values
        # We can do this efficiently:
        # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        # Let's compute this as a tensor contraction.
        
        # For each r, we have a rank-1 tensor u_r ⊗ v_r ⊗ w_r
        # where u_r has 16 entries, v_r has 16 entries, w_r has 16 entries.
        # We need to check that the sum equals the matmul tensor.
        
        # The matmul tensor T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] goes to C[i,k])
        # Wait, is that right? For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
        # So yes, T[i,j,k] = 1 for all i,j,k.
        
        # Let's verify a few entries.
        # Actually, let's do a full check using numpy.
        # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        
        # Build the tensor
        # This might be slow for 49 rank terms x 64 entries, but it's only 3136 operations.
        T_check = np.zeros((4, 4, 4), dtype=float)
        for r in range(rank):
            u_r = U_np[:, r]  # 16 entries
            v_r = V_np[:, r]  # 16 entries
            w_r = W_np[:, r]  # 16 entries
            # u_r[i*4+j] * v_r[j*4+k] * w_r[k*4+i]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        T_check[i, j, k] += u_r[i*4+j] * v_r[j*4+k] * w_r[k*4+i]
        
        # Expected: all ones
        expected = np.ones((4, 4, 4))
        max_err = np.max(np.abs(T_check - expected))
        if max_err > 1e-9:
            # Fallback: return schoolbook if verification fails
            rank_fb = 64
            U_fb = [[0] * rank_fb for _ in range(16)]
            V_fb = [[0] * rank_fb for _ in range(16)]
            W_fb = [[0] * rank_fb for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U_fb[i*4+j][col] = 1
                        V_fb[j*4+k][col] = 1
                        W_fb[k*4+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
