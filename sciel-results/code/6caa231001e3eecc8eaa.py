"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    # Direct-sum decomposition exploiting the column-independence of the 2x2 by 2x15 product.
    # The tensor M_{2,2,15} is the direct sum of 15 copies of M_{2,2,1}.
    # For each 2x2 by 2x1 block, we use the known minimal rank-4 decomposition
    # (Strassen's algorithm on 2x2x1, which is a direct sum of two independent
    # 2x2x1 problems, each of which has rank 2? No, M_{2,2,1} has rank 4? Let's verify.)
    #
    # Actually, M_{2,2,1} corresponds to computing C_{i,k} = sum_j A_{i,j} B_{j,k} for a single column k.
    # C is 2x1. C_0 = A_00*B_0k + A_01*B_1k, C_1 = A_10*B_0k + A_11*B_1k.
    # This is two independent dot products of length 2.
    # The rank of the dot-product tensor of length 2 is 2.
    # Since the two outputs C_0 and C_1 share the same input B_0k, B_1k but have different A rows,
    # can we share anything?
    # The tensor is: sum_{i,j} A_{i,j} B_{j,k} C_{i,k}.
    # For fixed k, this is the tensor for matrix-vector multiplication 2x2 * 2x1.
    # The rank of M_{2,2,1} is 4. (It's the direct sum of two rank-2 tensors? No, the B indices are shared.)
    # Let's use the standard schoolbook for the 2x2x1 part: rank 4.
    # So total rank is 15 * 4 = 60.
    #
    # Wait, the previous response claimed Strassen on pairs of columns gives rank 53.
    # Let's check if Strassen applies to 2x2 by 2x2.
    # Strassen computes 2x2 * 2x2 with rank 7.
    # If we take two columns of B, say column k and k+1, we have a 2x2 matrix B_pair.
    # We compute A (2x2) * B_pair (2x2) = C_pair (2x2).
    # This uses rank 7.
    # We have 15 columns. We can form 7 pairs and 1 single column.
    # 7 pairs * rank 7 = 49.
    # 1 single column * rank 4 = 4.
    # Total rank = 53.
    # This is a valid improvement over 60.
    #
    # Let's verify the Strassen decomposition for 2x2 by 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-c)(e+g)
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # We need to map this to U, V, W.
    # For each pair of columns (2k, 2k+1), we create 7 terms.
    # For the last column (28? no, 14), we create 4 terms.
    #
    # Let's define the indices.
    # U has shape (2*2, rank) = (4, rank). Indices: (i, j) -> i*m + j.
    # V has shape (2*15, rank) = (30, rank). Indices: (j, k) -> j*p + k.
    # W has shape (15*2, rank) = (30, rank). Indices: (k, i) -> k*n + i.
    #
    # For a pair (k, k+1), the output columns are 2k and 2k+1.
    # The input B columns are 2k and 2k+1.
    #
    # Let's implement this.
    #
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []
    # We will build the lists column by column (in the rank dimension).
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        for i in range(n*m):
            U.append(u_vec[i])
        for i in range(m*p):
            V.append(v_vec[i])
        for i in range(p*n):
            W.append(w_vec[i])
        rank += 1

    # We'll use 0-indexed columns for B and C.
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)
    # Single: 14
    
    for pair_idx in range(7):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # Strassen terms for 2x2 by 2x2
        # A is 2x2, B is 2x2 (columns k0, k1)
        # m1 = (A00+A11) * (B0k0+B1k1)  -- Wait, Strassen mixes rows and columns.
        # Standard Strassen for A*B:
        # m1 = (a+d)(e+h) = (A00+A11)(B00+B11) where B00 is B[0][k0], B11 is B[1][k1]
        # This mixes A's diagonal with B's diagonal across the pair.
        # Let's carefully map the indices.
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,k0), (0,k1), (1,k0), (1,k1)
        
        # m1: (A00+A11) * (B0k0+B1k1)
        u1 = [0,0,0,0]; u1[0] = 1; u1[3] = 1
        v1 = [0]*30; v1[0*15+k0] = 1; v1[1*15+k1] = 1
        w1 = [0]*30
        # C00 = m1 - m4 + m5 - m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        # So m1 contributes to C00 and C11.
        w1[k0*n+0] = 1   # C[k0, 0]
        w1[k1*n+1] = 1   # C[k1, 1]
        add_term(u1, v1, w1)
        
        # m2: (A10+A11) * B0k0  -- (c+d)*e
        u2 = [0,0,0,0]; u2[2] = 1; u2[3] = 1
        v2 = [0]*30; v2[0*15+k0] = 1
        w2 = [0]*30
        # C10 = m2 + m4, C11 = m1 - m2 + m3 + m6
        w2[k0*n+1] = 1   # C[k0, 1]
        w2[k1*n+1] = -1  # C[k1, 1]
        add_term(u2, v2, w2)
        
        # m3: A00 * (B0k1-B1k1)  -- a*(f-h)
        u3 = [0,0,0,0]; u3[0] = 1
        v3 = [0]*30; v3[0*15+k1] = 1; v3[1*15+k1] = -1
        w3 = [0]*30
        # C01 = m3 + m5, C11 = m1 - m2 + m3 + m6
        w3[k1*n+0] = 1   # C[k1, 0]
        w3[k1*n+1] = 1   # C[k1, 1]
        add_term(u3, v3, w3)
        
        # m4: A11 * (B1k0-B0k0)  -- d*(g-e)
        u4 = [0,0,0,0]; u4[3] = 1
        v4 = [0]*30; v4[1*15+k0] = 1; v4[0*15+k0] = -1
        w4 = [0]*30
        # C00 = m1 - m4 + m5 - m7, C10 = m2 + m4
        w4[k0*n+0] = -1  # C[k0, 0]
        w4[k0*n+1] = 1   # C[k0, 1]
        add_term(u4, v4, w4)
        
        # m5: (A00+A01) * B1k1  -- (a+b)*h
        u5 = [0,0,0,0]; u5[0] = 1; u5[1] = 1
        v5 = [0]*30; v5[1*15+k1] = 1
        w5 = [0]*30
        # C00 = m1 - m4 + m5 - m7, C01 = m3 + m5
        w5[k0*n+0] = 1   # C[k0, 0]
        w5[k1*n+0] = 1   # C[k1, 0]
        add_term(u5, v5, w5)
        
        # m6: (A10-A11) * (B0k1+B1k1)  -- (c-d)*(f+g)
        u6 = [0,0,0,0]; u6[2] = 1; u6[3] = -1
        v6 = [0]*30; v6[0*15+k1] = 1; v6[1*15+k1] = 1
        w6 = [0]*30
        # C11 = m1 - m2 + m3 + m6
        w6[k1*n+1] = 1   # C[k1, 1]
        add_term(u6, v6, w6)
        
        # m7: (A01-A10) * (B0k0+B1k0)  -- (b-c)*(e+g)
        u7 = [0,0,0,0]; u7[1] = 1; u7[2] = -1
        v7 = [0]*30; v7[0*15+k0] = 1; v7[1*15+k0] = 1
        w7 = [0]*30
        # C00 = m1 - m4 + m5 - m7
        w7[k0*n+0] = -1  # C[k0, 0]
        add_term(u7, v7, w7)
        
    # Last column: 14
    k = 14
    # Schoolbook for 2x2 by 2x1
    # C0 = A00*B0k + A01*B1k
    # C1 = A10*B0k + A11*B1k
    # 4 terms
    
    # T1: A00 * B0k
    u = [0,0,0,0]; u[0] = 1
    v = [0]*30; v[0*15+k] = 1
    w = [0]*30; w[k*n+0] = 1
    add_term(u, v, w)
    
    # T2: A01 * B1k
    u = [0,0,0,0]; u[1] = 1
    v = [0]*30; v[1*15+k] = 1
    w = [0]*30; w[k*n+0] = 1
    add_term(u, v, w)
    
    # T3: A10 * B0k
    u = [0,0,0,0]; u[2] = 1
    v = [0]*30; v[0*15+k] = 1
    w = [0]*30; w[k*n+1] = 1
    add_term(u, v, w)
    
    # T4: A11 * B1k
    u = [0,0,0,0]; u[3] = 1
    v = [0]*30; v[1*15+k] = 1
    w = [0]*30; w[k*n+1] = 1
    add_term(u, v, w)
    
    # Convert lists to matrices
    nU = n*m
    nV = m*p
    nW = p*n
    U_mat = [U[i::rank] for i in range(nU)]
    V_mat = [V[i::rank] for i in range(nV)]
    W_mat = [W[i::rank] for i in range(nW)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
