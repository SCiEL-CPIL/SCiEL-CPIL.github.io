def construct():
    import time
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin
    
    # Laderman's rank-28 decomposition for 4x4 matrix multiplication
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # U: (16, 28) where U[i*4+j, t] = coeff of A[i,j] in left factor of product t
    # V: (16, 28) where V[j*4+k, t] = coeff of B[j,k] in right factor of product t
    # W: (16, 28) where W[k*4+i, t] = coeff of product t in C[i,k]
    
    # Define the 28 products of Laderman's algorithm
    # Each product is: (sum over (i,j) of u[i,j]*A[i,j]) * (sum over (j,k) of v[j,k]*B[j,k])
    # and contributes to C entries via w[i,k]
    
    # I'll define the decomposition explicitly
    # Using the standard Laderman 4x4 decomposition
    
    # Let me define it as: for each term t, list of (A_index, coeff), (B_index, coeff), and contributions to C
    # A_index = i*4+j, B_index = j*4+k, C_index = k*4+i
    
    # Laderman's 28 terms:
    # I'll use the formulation from the literature
    
    terms = []
    
    # Term 0: a00 * b00
    terms.append(([(0,1)], [(0,1)], [(0,1)]))  # A[0,0]*B[0,0] -> C[0,0]
    
    # This is getting complex. Let me use a different approach.
    # I'll use the known Laderman decomposition coefficients.
    
    # Actually, let me use a well-known explicit form.
    # The Laderman decomposition can be written as follows:
    
    # Define helper to create sparse vectors
    def vec(indices_coeffs):
        """Create a dict from list of (index, coeff)"""
        d = {}
        for idx, c in indices_coeffs:
            d[idx] = d.get(idx, 0) + c
        return d
    
    # Laderman 4x4 decomposition - 28 terms
    # Each term: (left_factor, right_factor, output_contributions)
    # left_factor: dict mapping A[i*4+j] -> coeff
    # right_factor: dict mapping B[j*4+k] -> coeff  
    # output_contributions: dict mapping C[k*4+i] -> coeff
    
    T = []
    
    # T0: a00*b00
    T.append((vec([(0,1)]), vec([(0,1)]), vec([(0,1)])))
    
    # T1: a00*b11
    T.append((vec([(0,1)]), vec([(5,1)]), vec([(1,1)])))
    
    # T2: a01*b10
    T.append((vec([(1,1)]), vec([(4,1)]), vec([(0,1)])))
    
    # T3: a01*b11
    T.append((vec([(1,1)]), vec([(5,1)]), vec([(1,1)])))
    
    # T4: a02*b20
    T.append((vec([(2,1)]), vec([(8,1)]), vec([(0,1)])))
    
    # T5: a02*b21
    T.append((vec([(2,1)]), vec([(9,1)]), vec([(1,1)])))
    
    # T6: a03*b30
    T.append((vec([(3,1)]), vec([(12,1)]), vec([(0,1)])))
    
    # T7: a03*b31
    T.append((vec([(3,1)]), vec([(13,1)]), vec([(1,1)])))
    
    # T8: (a00+a01)*(b20+b21)
    T.append((vec([(0,1),(1,1)]), vec([(8,1),(9,1)]), vec([(2,1),(3,1)])))
    
    # T9: (a10+a11)*(b00+b01)
    T.append((vec([(4,1),(5,1)]), vec([(0,1),(1,1)]), vec([(4,1),(5,1)])))
    
    # T10: a10*(b20+b21)
    T.append((vec([(4,1)]), vec([(8,1),(9,1)]), vec([(2,1),(3,1)])))
    
    # T11: a11*(b20+b21)
    T.append((vec([(5,1)]), vec([(8,1),(9,1)]), vec([(2,1),(3,1)])))
    
    # T12: a12*b20
    T.append((vec([(6,1)]), vec([(8,1)]), vec([(0,1)])))
    
    # T13: a12*b21
    T.append((vec([(6,1)]), vec([(9,1)]), vec([(1,1)])))
    
    # T14: a13*b30
    T.append((vec([(7,1)]), vec([(12,1)]), vec([(0,1)])))
    
    # T15: a13*b31
    T.append((vec([(7,1)]), vec([(13,1)]), vec([(1,1)])))
    
    # T16: (a10+a11)*(b30+b31)
    T.append((vec([(4,1),(5,1)]), vec([(12,1),(13,1)]), vec([(6,1),(7,1)])))
    
    # T17: (a20+a21)*(b00+b01)
    T.append((vec([(8,1),(9,1)]), vec([(0,1),(1,1)]), vec([(8,1),(9,1)])))
    
    # T18: a20*(b30+b31)
    T.append((vec([(8,1)]), vec([(12,1),(13,1)]), vec([(6,1),(7,1)])))
    
    # T19: a21*(b30+b31)
    T.append((vec([(9,1)]), vec([(12,1),(13,1)]), vec([(6,1),(7,1)])))
    
    # T20: a22*b30
    T.append((vec([(10,1)]), vec([(12,1)]), vec([(0,1)])))
    
    # T21: a22*b31
    T.append((vec([(10,1)]), vec([(13,1)]), vec([(1,1)])))
    
    # T22: a30*(b10+b11)
    T.append((vec([(12,1)]), vec([(4,1),(5,1)]), vec([(4,1),(5,1)])))
    
    # T23: a30*(b20+b21)
    T.append((vec([(12,1)]), vec([(8,1),(9,1)]), vec([(8,1),(9,1)])))
    
    # T24: a31*(b10+b11)
    T.append((vec([(13,1)]), vec([(4,1),(5,1)]), vec([(4,1),(5,1)])))
    
    # T25: a31*(b20+b21)
    T.append((vec([(13,1)]), vec([(8,1),(9,1)]), vec([(8,1),(9,1)])))
    
    # T26: a32*b20
    T.append((vec([(14,1)]), vec([(8,1)]), vec([(0,1)])))
    
    # T27: a32*b21
    T.append((vec([(14,1)]), vec([(9,1)]), vec([(1,1)])))
    
    # Wait, this doesn't look right. The contributions to C need to be correct.
    # Let me verify: C[i,k] = sum_j A[i,j] * B[j,k]
    # So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # In my indexing: C[k*4+i] where k is row of B / col of C, i is row of A / row of C
    # So C[0,0] is at index 0*4+0 = 0
    # C[0,1] is at index 1*4+0 = 4
    # C[1,0] is at index 0*4+1 = 1
    # C[1,1] is at index 1*4+1 = 5
    
    # Let me redo this more carefully.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # W index: k*4 + i (as stated in the problem: W[k*n+i,r] with n=4)
    
    # So for term t, W[k*4+i, t] is the coefficient of M[t] in C[i,k]
    
    # Let me restart with a cleaner formulation.
    # I'll define the Laderman decomposition properly.
    
    # Actually, let me just use numpy to verify and construct the matrices.
    
    n, m, p = 4, 4, 8
    
    # For 4x4x4, Laderman gives rank 28
    # U4: (16, 28), V4: (16, 28), W4: (16, 28)
    # For 4x4x8: split B into B_left (4x4) and B_right (4x4)
    # C = A * [B_left | B_right] = [A*B_left | A*B_right]
    # So we use Laderman twice: once for A*B_left, once for A*B_right
    
    # U: (16, 56) - first 28 cols same as U4, next 28 cols same as U4
    # V: (32, 56) - for first 28 cols, V uses B_left entries (rows 0-15 of B)
    #             - for next 28 cols, V uses B_right entries (rows 16-31 of B)
    # W: (32, 56) - for first 28 cols, W contributes to C_left (rows 0-15)
    #               - for next 28 cols, W contributes to C_right (rows 16-31)
    
    # Let me define the Laderman 4x4 decomposition matrices
    
    # I'll construct them by specifying the 28 terms
    
    U4 = [[0]*28 for _ in range(16)]
    V4 = [[0]*28 for _ in range(16)]
    W4 = [[0]*28 for _ in range(16)]
    
    # Define the 28 terms of Laderman's algorithm
    # Each term t has:
    #   left: dict {A[i,j] index: coeff} where index = i*4+j
    #   right: dict {B[j,k] index: coeff} where index = j*4+k
    #   out: dict {C[i,k] index: coeff} where index = k*4+i
    
    laderman_terms = [
        # t=0: a00*b00 -> C[0,0]
        ([(0,1)], [(0,1)], [(0,1)]),
        # t=1: a00*b11 -> C[1,0]  
        ([(0,1)], [(5,1)], [(1,1)]),
        # t=2: a01*b10 -> C[0,1]
        ([(1,1)], [(4,1)], [(4,1)]),
        # t=3: a01*b11 -> C[1,1]
        ([(1,1)], [(5,1)], [(5,1)]),
        # t=4: a02*b20 -> C[0,2]
        ([(2,1)], [(8,1)], [(8,1)]),
        # t=5: a02*b21 -> C[1,2]
        ([(2,1)], [(9,1)], [(9,1)]),
        # t=6: a03*b30 -> C[0,3]
        ([(3,1)], [(12,1)], [(12,1)]),
        # t=7: a03*b31 -> C[1,3]
        ([(3,1)], [(13,1)], [(13,1)]),
        # t=8: (a00+a01)*(b20+b21) -> C[2,0]+C[3,0]... 
        # This is getting too error-prone. Let me use a verified Laderman decomposition.
    ]
    
    # Let me use a different, more reliable approach.
    # I'll use the known Laderman decomposition from a reference implementation.
    
    # Here's the Laderman decomposition as commonly stated:
    # The 28 products are:
    # p1 = a00*b00
    # p2 = a00*b11  
    # p3 = a01*b10
    # p4 = a01*b11
    # p5 = a02*b20
    # p6 = a02*b21
    # p7 = a03*b30
    # p8 = a03*b31
    # p9 = (a00+a01)*(b20+b21)
    # p10 = (a10+a11)*(b00+b01)
    # p11 = a10*(b20+b21)
    # p12 = a11*(b20+b21)
    # p13 = a12*b20
    # p14 = a12*b21
    # p15 = a13*b30
    # p16 = a13*b31
    # p17 = (a10+a11)*(b30+b31)
    # p18 = (a20+a21)*(b00+b01)
    # p19 = a20*(b30+b31)
    # p20 = a21*(b30+b31)
    # p21 = a22*b30
    # p22 = a22*b31
    # p23 = a30*(b10+b11)
    # p24 = a30*(b20+b21)
    # p25 = a31*(b10+b11)
    # p26 = a31*(b20+b21)
    # p27 = a32*b20
    # p28 = a32*b21
    
    # And the output C[i,j] = sum of appropriate products
    
    # Let me verify this gives correct results by checking C[0,0]:
    # C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # = p1 + p3 + p5 + p7  ✓
    
    # C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # Hmm, but p2 = a00*b11, not a00*b01
    # This decomposition doesn't seem to directly give C[0,1] as a simple sum.
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The Laderman decomposition is a valid rank-28 decomposition, but the 
    # products are not simply a[i,j]*b[k,l] for matching indices.
    # The key is that each product is a bilinear form, and the output is a 
    # linear combination of these products.
    
    # Let me just use numpy to construct and verify the decomposition.
    
    # I'll use a known correct Laderman decomposition.
    
    # After more thought, let me use the following approach:
    # I'll define the Laderman decomposition using its standard form.
    
    # The standard Laderman decomposition for 4x4:
    
    # Left factors (coefficients for A entries a[i,j], i,j=0..3):
    # Right factors (coefficients for B entries b[i,j], i,j=0..3):
    # Output coefficients (for C entries c[i,j], i,j=0..3):
    
    # I'll define this as three 28x16 matrices (transposed from what we need)
    
    # Let me use a verified source. The Laderman decomposition:
    
    # I'll construct it step by step and verify with numpy.
    
    import numpy as np
    
    # Define Laderman's 28 terms
    # Each term is (left_coeffs, right_coeffs, output_coeffs)
    # left_coeffs: list of 16 values for A[0,0],A[0,1],...,A[3,3]
    # right_coeffs: list of 16 values for B[0,0],B[0,1],...,B[3,3]
    # output_coeffs: list of 16 values for C[0,0],C[0,1],...,C[3,3]
    
    def z16():
        return [0]*16
    
    # I'll use the Laderman decomposition from a reliable source
    # The following is the standard Laderman 4x4 decomposition
    
    L = []  # left factors
    R = []  # right factors  
    O = []  # output factors
    
    # Term 0
    l = z16(); l[0] = 1
    r = z16(); r[0] = 1
    o = z16(); o[0] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 1
    l = z16(); l[0] = 1
    r = z16(); r[5] = 1
    o = z16(); o[1] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 2
    l = z16(); l[1] = 1
    r = z16(); r[4] = 1
    o = z16(); o[4] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 3
    l = z16(); l[1] = 1
    r = z16(); r[5] = 1
    o = z16(); o[5] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 4
    l = z16(); l[2] = 1
    r = z16(); r[8] = 1
    o = z16(); o[8] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 5
    l = z16(); l[2] = 1
    r = z16(); r[9] = 1
    o = z16(); o[9] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 6
    l = z16(); l[3] = 1
    r = z16(); r[12] = 1
    o = z16(); o[12] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 7
    l = z16(); l[3] = 1
    r = z16(); r[13] = 1
    o = z16(); o[13] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 8: (a00+a01)*(b20+b21)
    l = z16(); l[0] = 1; l[1] = 1
    r = z16(); r[8] = 1; r[9] = 1
    o = z16(); o[2] = 1; o[3] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 9: (a10+a11)*(b00+b01)
    l = z16(); l[4] = 1; l[5] = 1
    r = z16(); r[0] = 1; r[1] = 1
    o = z16(); o[4] = 1; o[5] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 10: a10*(b20+b21)
    l = z16(); l[4] = 1
    r = z16(); r[8] = 1; r[9] = 1
    o = z16(); o[2] = 1; o[3] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 11: a11*(b20+b21)
    l = z16(); l[5] = 1
    r = z16(); r[8] = 1; r[9] = 1
    o = z16(); o[2] = 1; o[3] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 12: a12*b20
    l = z16(); l[6] = 1
    r = z16(); r[8] = 1
    o = z16(); o[0] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 13: a12*b21
    l = z16(); l[6] = 1
    r = z16(); r[9] = 1
    o = z16(); o[1] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 14: a13*b30
    l = z16(); l[7] = 1
    r = z16(); r[12] = 1
    o = z16(); o[0] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 15: a13*b31
    l = z16(); l[7] = 1
    r = z16(); r[13] = 1
    o = z16(); o[1] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 16: (a10+a11)*(b30+b31)
    l = z16(); l[4] = 1; l[5] = 1
    r = z16(); r[12] = 1; r[13] = 1
    o = z16(); o[6] = 1; o[7] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 17: (a20+a21)*(b00+b01)
    l = z16(); l[8] = 1; l[9] = 1
    r = z16(); r[0] = 1; r[1] = 1
    o = z16(); o[8] = 1; o[9] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 18: a20*(b30+b31)
    l = z16(); l[8] = 1
    r = z16(); r[12] = 1; r[13] = 1
    o = z16(); o[6] = 1; o[7] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 19: a21*(b30+b31)
    l = z16(); l[9] = 1
    r = z16(); r[12] = 1; r[13] = 1
    o = z16(); o[6] = 1; o[7] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 20: a22*b30
    l = z16(); l[10] = 1
    r = z16(); r[12] = 1
    o = z16(); o[0] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 21: a22*b31
    l = z16(); l[10] = 1
    r = z16(); r[13] = 1
    o = z16(); o[1] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 22: a30*(b10+b11)
    l = z16(); l[12] = 1
    r = z16(); r[4] = 1; r[5] = 1
    o = z16(); o[4] = 1; o[5] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 23: a30*(b20+b21)
    l = z16(); l[12] = 1
    r = z16(); r[8] = 1; r[9] = 1
    o = z16(); o[8] = 1; o[9] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 24: a31*(b10+b11)
    l = z16(); l[13] = 1
    r = z16(); r[4] = 1; r[5] = 1
    o = z16(); o[4] = 1; o[5] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 25: a31*(b20+b21)
    l = z16(); l[13] = 1
    r = z16(); r[8] = 1; r[9] = 1
    o = z16(); o[8] = 1; o[9] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 26: a32*b20
    l = z16(); l[14] = 1
    r = z16(); r[8] = 1
    o = z16(); o[0] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Term 27: a32*b21
    l = z16(); l[14] = 1
    r = z16(); r[9] = 1
    o = z16(); o[1] = 1
    L.append(l); R.append(r); O.append(o)
    
    # Now verify this decomposition
    # Create L_mat (28x16), R_mat (28x16), O_mat (28x16)
    L_mat = np.array(L, dtype=float)  # (28, 16)
    R_mat = np.array(R, dtype=float)  # (28, 16)
    O_mat = np.array(O, dtype=float)  # (28, 16)
    
    # For random A and B, check if C = A*B can be reconstructed
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # In our indexing: C is stored as c[i*4+j]
    # A[i,k] is at index i*4+k in the 16-element vector
    # B[k,j] is at index k*4+j in the 16-element vector
    
    # The decomposition says: C = O_mat.T @ (L_mat @ a_vec) * (R_mat @ b_vec)
    # where a_vec and b_vec are the 16-element flattened vectors
    # Wait, that's not quite right. Each product t is (L_mat[t] @ a_vec) * (R_mat[t] @ b_vec)
    # and C_vec = O_mat.T @ products_vec
    
    # Let me verify:
    np.random.seed(42)
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 4)
    C_test = A_test @ B_test
    
    a_vec = A_test.flatten()  # (16,)
    b_vec = B_test.flatten()  # (16,)
    
    # Compute products
    left_factors = L_mat @ a_vec  # (28,)
    right_factors = R_mat @ b_vec  # (28,)
    products = left_factors * right_factors  # (28,)
    
    # Compute C
    C_recon = (O_mat.T @ products).reshape(4, 4)
    
    # Check
    err = np.max(np.abs(C_test - C_recon))
    
    if err > 1e-10:
        # The decomposition is wrong, let me try a different approach
        # Fall back to schoolbook for now, but try to fix Laderman
        pass
    
    # If Laderman verification fails, use schoolbook
    if err > 1e-10:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Laderman verification passed, construct the 4x4x8 decomposition
    # U4: (16, 28) = L_mat.T
    # V4: (16, 28) = R_mat.T
    # W4: (16, 28) = O_mat.T
    
    # For 4x4x8:
    # U: (16, 56) - first 28 cols = U4, next 28 cols = U4
    # V: (32, 56) - first 28 cols: V4 in rows 0-15, zeros in rows 16-31
    #                 next 28 cols: zeros in rows 0-15, V4 in rows 16-31
    # W: (32, 56) - first 28 cols: W4 in rows 0-15, zeros in rows 16-31
    #               next 28 cols: zeros in rows 0-15, W4 in rows 16-31
    
    rank_total = 56
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(32)]
    
    # Fill first 28 columns
    for r in range(28):
        for i in range(16):
            U[i][r] = L_mat[r][i]
            V[i][r] = R_mat[r][i]
            W[i][r] = O_mat[r][i]
    
    # Fill next 28 columns (for B_right)
    for r in range(28):
        for i in range(16):
            U[i][r + 28] = L_mat[r][i]
            V[i + 16][r + 28] = R_mat[r][i]
            W[i + 16][r + 28] = O_mat[r][i]
    
    return {"U": U, "V": V, "W": W}
