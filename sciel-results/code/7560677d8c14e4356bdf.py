"""Matrix multiplication 4,4,8 via Strassen-based construction."""
DIMENSIONS = [4, 4, 8]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2 tensors. Each rank-1 is (u, v, w) with u,v,w in R^4 (flattened 2x2)
    # u, v are the coefficient vectors for X and Y (2x2), w is the coefficient vector for Z (2x2)
    # Convention: index = row*2 + col
    # X00=0, X01=1, X10=2, X11=3
    strassen_tensors = [
        (
            [1, 0, 0, 1],  # X00 + X11
            [1, 0, 0, 1],  # Y00 + Y11
            [1, 0, 1, 0],  # Z00 + Z10  (col 0)
        ),
        (
            [0, 0, 1, 1],  # X10 + X11
            [1, 0, 0, 0],  # Y00
            [0, 0, 1, 0],  # Z10
        ),
        (
            [1, 0, 0, 0],  # X00
            [0, 1, 0, 1],  # Y01 + Y11
            [1, 0, 0, 0],  # Z00
        ),
        (
            [0, 0, 0, 1],  # X11
            [0, 1, 0, -1], # Y11 - Y01
            [0, 0, 0, 1],  # Z11
        ),
        (
            [0, 1, 0, 0],  # X01
            [1, 0, 1, 0],  # Y00 + Y10
            [0, 1, 0, 0],  # Z01
        ),
        (
            [0, 0, 1, 0],  # X10
            [1, 1, 0, 0],  # Y10 + Y00? Let me re-derive
            # Standard Strassen:
            # M5 = (X10 - X00)(Y11 + X01)
            # Actually let me use the standard 7 formulas:
            # M1 = (X00 + X11)(Y00 + Y11)
            # M2 = (X10 + X11)Y00
            # M3 = X00(Y01 - Y11)
            # M4 = X11(Y10 - Y00)
            # M5 = (X00 + X01)Y11
            # M6 = (X10 - X00)(Y00 + Y10)
            # M7 = (X11 - X10)(Y10 + Y11)
            # Then:
            # Z00 = M1 + M4 - M5 - M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M7
        ),
    ]

    # Let me define the 7 Strassen rank-1 terms properly:
    # M1: (X00+X11)(Y00+Y11) -> Z00 += 1, Z10 += 1, Z01 += ... 
    # Actually each M contributes to multiple Z entries.
    # Standard form: each of the 7 is a scalar product, and Z is a linear combination.
    # For rank-1 tensor form, each M_s contributes one rank-1 tensor where
    # u = coeff of X entries, v = coeff of Y entries, w = coeff of Z entries that this product contributes to.
    # But a single product M_s = (sum a_i X_i)(sum b_j Y_j) contributes a*b to specific Z entries.
    # In rank-1 tensor form, we can have each M_s contribute separately to each Z entry it affects.
    # That would give 7*4 = 28 rank-1 terms per 2x2 multiplication, which is worse.
    # 
    # Actually, the correct rank-1 decomposition of 2x2 matrix multiplication has rank 7, where
    # each rank-1 term is (u_s, v_s, w_s) and the product is sum_s M_s = sum_s (u_s . X)(v_s . Y)(w_s . Z^T)
    # This means: for each s, (u_s . X)(v_s . Y) contributes to Z via w_s.
    # i.e., Z[w_s] += (u_s . X)(v_s . Y) for each w in supp(w_s).
    # 
    # The 7 terms for 2x2 (using standard indexing x00,x01,x10,x11):
    # Let me use the formulation where:
    # M1 = (x00+x11)(y00+y11), contributes: z00+=M1, z10+=M1, z11+=M1? No.
    
    # Let me just directly list the 7 rank-1 decompositions from literature:
    # Using the fact that 2x2x2 has rank 7. One known decomposition:
    # T1 = (e00+e11) ⊗ (e00+e11) ⊗ (e00)  ... this isn't right either.
    
    # Let me think differently. The tensor T_{ijk} = 1 if k=i+j mod... no, for matrix mult it's sum_j A_ij B_jk = C_ik.
    # T[i,k,j] = δ_ij * δ_kj... 
    
    # For 2x2: C_ik = sum_j A_ij B_jk
    # The 7 rank-1 tensors (from standard references) using flattening (i,j) index 0..3:
    # I'll use the well-known decomposition. Let me hardcode the 7 triples.
    
    def strassen_2x2():
        # Returns list of 7 (u, v, w) vectors of length 4 each
        # u[i,j], v[j,k], w[i,k] indexed flat
        # Based on: https://en.wikipedia.org/wiki/Strassen_algorithm
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A11-A10)(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Each M_s is a scalar. The rank-1 tensor for M_s has:
        # u_s = coeff vector of A in M_s
        # v_s = coeff vector of B in M_s
        # w_s = coefficient by which M_s contributes to each C entry
        
        terms = []
        # M1: A coeff [1,0,0,1], B coeff [1,0,0,1], contributes +1 to C00, +1 to C10, +1 to C11? 
        # C00 = ...+M1..., C10 = ...+... no, C10 = M2+M4. 
        # M1 appears in C00 (+1) and C11 (-1... wait C11 = M1-M2+M3+M7)
        # So M1: +1 to C00, +1 to C11? Let me recheck.
        # Standard: 
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So each M_s contributes to multiple C entries with different signs.
        # In rank-1 tensor decomposition, the third factor w_s must capture all contributions.
        # M1: w = [1, 0, 0, 1]  (C00=idx0, C01=idx1, C10=idx2, C11=idx3)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2: A=[0,0,1,1], B=[1,0,0,0], w=[0,0,1,-1] (C10 +1, C11 -1)
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # M3: A=[1,0,0,0], B=[0,1,0,-1], w=[0,1,0,1] (C01 +1, C11 +1)
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        # M4: A=[0,0,0,1], B=[0,0,1,-1], w=[1,0,1,0] (C00 +1, C10 +1)
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        # M5: A=[1,1,0,0], B=[0,0,0,1], w=[-1,1,0,0] (C00 -1, C01 +1)
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # M6: A=[-1,0,1,0], B=[1,0,1,0], w=[0,0,0,0] -- M6 doesn't appear in any C!
        # Wait, M6 doesn't appear? That can't be right. Let me recheck the standard formulas.
        
        # I think I have the wrong set. Let me use a verified decomposition.
        # The rank of 2x2 matrix multiplication is 7. Here's a correct explicit decomposition.
        # Using indices 0=00, 1=01, 2=10, 3=11:
        
        # Actually, I realize the standard Strassen 7 formulas DO all appear in C. Let me recheck.
        # I had C11 = M1 - M2 + M3 + M7. Where does M6 go?
        # M6 is not used? That's suspicious. Let me look up again.
        
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00)
        # M3 = (A00)(B01 - B11)
        # M4 = (A11)(B10 - B00)
        # M5 = (A00 + A01)(B11)
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A11 - A10)(B10 + B11)
        #
        # C00 = M1 - M5 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # So M6 contributes to C00. Let me redo:
        terms = []
        # M1: A=[1,0,0,1], B=[1,0,0,1], C00 +1, C11 +1
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2: A=[0,0,1,1], B=[1,0,0,0], C10 +1, C11 -1
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        # M3: A=[1,0,0,0], B=[0,1,0,-1], C01 +1, C11 +1
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
        # M4: A=[0,0,0,1], B=[0,0,1,-1], C10 +1
        terms.append(([0,0,0,1], [0,0,1,-1], [0,0,1,0]))
        # M5: A=[1,1,0,0], B=[0,0,0,1], C00 -1, C01 +1
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # M6: A=[-1,0,1,0], B=[1,0,1,0], C00 +1
        terms.append(([-1,0,1,0], [1,0,1,0], [1,0,0,0]))
        # M7: A=[0,0,-1,1], B=[0,0,1,1], C00 -1, C11 +1
        terms.append(([0,0,-1,1], [0,0,1,1], [-1,0,0,1]))
        
        return terms
    
    # For 4x4x4, use Strassen recursively: 4x4 = 2x(2x2). 
    # But the 4x4x4 tensor has rank at most 7*7 = 49 by recursion.
    # However, constructing this explicitly is complex. Let me instead use a direct approach:
    # 
    # For 4x4 * 4x8, think of it as multiplying 4x4 by two 4x4 blocks.
    # Each 4x4*4x4 can be done with rank <= 49 (Strassen recursive).
    # Total rank <= 98.
    #
    # But we need explicit rank-1 tensors. Let me build the 4x4x4 decomposition using
    # the 2x2 Strassen tensors.
    
    def build_4x4x4_decomp():
        """Build rank-1 decomposition for 4x4 * 4x4 -> 4x4.
        Returns list of (U, V, W) vectors of length 16 each (flattened 4x4).
        """
        base = strassen_2x2()
        # 4x4 block structure: A[i,j] with i,j in {0,1,2,3}
        # Split into 2x2 blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right)
        # Each block is 2x2, with entries at positions:
        # A00: (0,0),(0,1),(1,0),(1,1) -> flat indices 0,1,4,5
        # A01: (0,2),(0,3),(1,2),(1,3) -> flat indices 2,3,6,7
        # A10: (2,0),(2,1),(3,0),(3,1) -> flat indices 8,9,12,13
        # A11: (2,2),(2,3),(3,2),(3,3) -> flat indices 10,11,14,15
        
        # For a 2x2 Strassen term (u, v, w) where u,v,w are length-4 vectors (indices 0..3 in 2x2 block),
        # we need to map to the 4x4 space.
        
        # Block mapping: block r,c (r in {0,1}, c in {0,1}) has 2x2 submatrix.
        # For A (input): block (r,c) occupies rows 2r..2r+1, cols 2c..2c+1
        # For B (input): block (r,c) occupies rows 2r..2r+1, cols 2c..2c+1
        # For C (output): block (r,c) occupies rows 2r..2r+1, cols 2c..2c+1
        
        def block_indices_4x4(block_r, block_c):
            """Return list of 4 flat indices (in row-major 4x4) for the 2x2 block at (block_r, block_c)."""
            idxs = []
            for dr in range(2):
                for dc in range(2):
                    row = 2*block_r + dr
                    col = 2*block_c + dc
                    idxs.append(row*4 + col)
            return idxs
        
        terms_4x4 = []
        for s in range(7):
            u2, v2, w2 = base[s]
            # u2 is a 2x2 block coefficient vector for the "A-side" of one 2x2 multiplication
            # In the 4x4 Strassen recursion, we apply the 2x2 algorithm to the 2x2 block structure.
            # This means: we treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
            # The Strassen formula operates on these blocks.
            # 
            # But wait, the standard Strassen 2x2 works on scalar entries. To use it on blocks,
            # we'd need the 2x2 block matrices to be commutative, which they aren't generally.
            # So naive block Strassen doesn't work for matrices unless the blocks commute.
            #
            # Alternative: apply 2x2 Strassen to individual entries, but that gives rank 7^4 = 2401 which is worse.
            #
            # Correct approach: The 4x4x4 multiplication tensor can be decomposed by viewing 4=2*2
            # and using the tensor product structure. The rank is at most rank(2x2x2)^2 = 49.
            # 
            # The tensor for 4x4x4 is (T_2 ⊗ T_2) where T_2 is the 2x2x2 multiplication tensor.
            # rank(T_2 ⊗ T_2) <= rank(T_2)^2 = 49.
            # 
            # To explicitly construct: if T_2 = sum_{s=1}^7 u_s ⊗ v_s ⊗ w_s, then
            # T_2 ⊗ T_2 = sum_{s,t=1}^{7,7} (u_s ⊗ u_t) ⊗ (v_s ⊗ v_t) ⊗ (w_s ⊗ w_t)
            # 
            # Where u_s ⊗ u_t means the tensor product of the vectors, giving a vector of length 4*4=16.
            # But we need to be careful about the indexing/ordering.
            
            # Let's define the vector product: if u2 is a vector of length 4 (representing a 2x2 matrix flat),
            # and we take u2 ⊗ u2 (Kronecker product), we get a vector of length 16.
            # The Kronecker product u2 ⊗ u2[i] = u2[i//4] * u2[i%4] ... no, 
            # numpy kdot(u, v) gives [u0*v0, u0*v1, ..., u1*v0, u1*v1, ...]
            
            pass
        
        # Let me use a more direct construction using NumPy's kron.
        import numpy as np
        
        base_np = [np.array(t, dtype=float) for t in base]
        u_list = [base_np[s][0] for s in range(7)]
        v_list = [base_np[s][1] for s in range(7)]
        w_list = [base_np[s][2] for s in range(7)]
        
        terms_4x4 = []
        for s in range(7):
            for t in range(7):
                u = np.kron(u_list[s], u_list[t])
                v = np.kron(v_list[s], v_list[t])
                w = np.kron(w_list[s], w_list[t])
                terms_4x4.append((list(u), list(v), list(w)))
        
        return terms_4x4
    
    decomp_4x4x4 = build_4x4x4_decomp()
    # This gives 49 terms for 4x4 * 4x4 -> 4x4
    
    # For 4x4 * 4x8: view B as two 4x4 blocks B[:, 0:4] and B[:, 4:8]
    # C[:, 0:4] = A * B[:, 0:4], C[:, 4:8] = A * B[:, 4:8]
    # Total rank: 49 + 49 = 98
    
    # Now build U, V, W matrices.
    # U: shape (n*m, rank) = (16, 98)
    # V: shape (m*p, rank) = (32, 98)
    # W: shape (p*n, rank) = (32, 98)
    
    # For the first half (columns 0-3 of C):
    # U entries come from A (16 entries, flattened 4x4)
    # V entries come from B_block0 (16 entries, flattened 4x4, which is B[:, 0:4])
    # W entries come from C_block0 (16 entries, flattened 4x4, which is C[:, 0:4])
    
    # For the second half (columns 4-7 of C):
    # U entries same A
    # V entries from B_block1 (16 entries, B[:, 4:8])
    # W entries from C_block1 (16 entries, C[:, 4:8])
    
    rank = 98
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(32)]
    
    import numpy as np
    
    # Map 4x4 flat index (row-major, index = i*4+j) to appropriate positions
    # For A (n*m = 4*4 = 16): index = i*4 + j, directly maps to U row
    # For B (m*p = 4*8 = 32): B[j,k] with j in 0..3, k in 0..7. 
    #   For block 0 (k=0..3): B_block0[j, k-0] -> flat in B_block0 is j*4+(k-0), maps to V row j*8+k
    #   For block 1 (k=4..7): B_block1[j, k-4] -> flat in B_block1 is j*4+(k-4), maps to V row j*8+k
    # For C (p*n = 8*4 = 32): C[k,i] with k in 0..7, i in 0..3.
    #   Wait, the convention is W[k*n+i, r]. So C is indexed as [k*n + i] where k is row in B (0..7) and i is col in A (0..3).
    #   So C has shape (8, 4) effectively, with entries C[k, i].
    #   For block 0 (k=0..3): C_block0[k, i] -> W row k*4 + i
    #   For block 1 (k=4..7): C_block1[k-4, i] -> W row k*4 + i
    
    # The 4x4x4 decomposition terms have vectors of length 16.
    # u_vec[alpha] corresponds to A[alpha//4, alpha%4]
    # v_vec[beta] corresponds to B_block[beta//4, beta%4]
    # w_vec[gamma] corresponds to C_block[gamma//4, gamma%4]
    
    for term_idx in range(49):
        u16, v16, w16 = decomp_4x4x4[term_idx]
        # First half
        col = term_idx
        for idx in range(16):
            U[idx][col] = u16[idx]
            # B_block0: flat idx = j*4 + kk where j in 0..3, kk in 0..3 -> V row = j*8 + kk
            j = idx // 4
            kk = idx % 4
            V[j*8 + kk][col] = v16[idx]
            # C_block0: flat idx = k*4 + i where k in 0..3, i in 0..3 -> W row = k*4 + i
            k = idx // 4
            i = idx % 4
            W[k*4 + i][col] = w16[idx]
        
        # Second half
        col2 = 49 + term_idx
        for idx in range(16):
            U[idx][col2] = u16[idx]
            j = idx // 4
            kk = idx % 4
            V[j*8 + 4 + kk][col2] = v16[idx]
            k = idx // 4
            i = idx % 4
            W[4*4 + k*4 + i][col2] = w16[idx]
    
    # Verify no NaN or inf
    import math
    for row in U:
        for val in row:
            if not math.isfinite(val):
                raise ValueError("Non-finite value in U")
    for row in V:
        for val in row:
            if not math.isfinite(val):
                raise ValueError("Non-finite value in V")
    for row in W:
        for val in row:
            if not math.isfinite(val):
                raise ValueError("Non-finite value in W")
    
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
