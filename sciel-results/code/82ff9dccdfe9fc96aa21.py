"""Program for 4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    # Use a rank-21 decomposition for 3x3 matrix multiplication.
    # This is a known minimal-ish decomposition.
    # The 7 known rank-21 decompositions for 3x3 are based on Strassen-like methods.
    # Here I use a specific one from the literature (e.g., from the work of Landsberg, etc.)
    # Actually, the minimum known rank for 3x3 is 19 (found by Bläser), but a simple rank-21 is easier.
    # Let me use a specific rank-21 decomposition.
    # 
    # A known rank-21 decomposition for 3x3:
    # Based on the 7-fold Strassen method or similar.
    # Actually, let me use a simple embedding approach.
    # 
    # Alternative: Use the rank-7 Strassen for 2x2 and pad to 3x3, giving rank 7*3=21 or similar.
    # Actually, if we embed 2x2 in 3x3, we get rank 7*3 = 21 if we use 3 copies? No.
    # 
    # Let me use a direct rank-21 decomposition from a known source.
    # 
    # Actually, let me try a different approach: use the rank-17 decomposition for 2x2? No, Strassen is rank 7.
    # 
    # Let me construct a 4x4 decomposition directly using a Kronecker product of two rank-7 Strassen decompositions,
    # but that gives rank 49.
    # 
    # Instead, let me use a more efficient method: 
    # The tensor for 4x4 = (2x2) ⊗ (2x2) can be decomposed with rank <= 7*7 = 49, but we can do better.
    # 
    # Actually, a known result: the rank of 4x4 matrix multiplication is at most 21*2 = 42? No.
    # 
    # Let me use the following: 
    # For 3x3, use a rank-19 decomposition (Bläser's). But implementing that is complex.
    # 
    # Simpler: use a rank-21 decomposition for 3x3, then embed in 4x4.
    # If we have a rank-21 decomposition for 3x3, we can embed it in 4x4 by padding, giving rank 21*4 = 84? No, that's worse.
    # 
    # Actually, embedding 3x3 in 4x4: if we have a rank-r decomposition for 3x3, we can get a rank-4r decomposition for 4x4 by using 4 copies (one for each "block" of the 4x4 that overlaps with 3x3). That's not efficient.
    # 
    # Better: use the fact that 4x4 can be decomposed using 2x2 blocks with a clever overlap.
    # 
    # Let me use a direct construction: 
    # Use the rank-7 Strassen decomposition for 2x2, and apply it to the four 2x2 blocks of the 4x4 matrix, but with overlapping to reduce rank.
    # 
    # Actually, the simplest improvement is to use a rank-49 decomposition (Kronecker product of two Strassen-7), which is already better than 64.
    # But the directive says to use a rank-21 decomposition for 3x3.
    # 
    # Let me implement a known rank-21 decomposition for 3x3.
    # From a paper: "A rank 21 decomposition for 3x3 matrix multiplication"
    # 
    # One such decomposition uses the following 21 triples (i,j,k) for the output, and linear forms for A and B.
    # 
    # Actually, let me use a simpler approach: 
    # I'll use the rank-7 Strassen for 2x2, and construct a 4x4 decomposition with rank 49.
    # This is already an improvement over 64.
    # 
    # But the directive specifically mentions using a rank-21 for 3x3. Let me try to implement that.
    # 
    # Known rank-21 decomposition for 3x3 (from Bläser or similar):
    # 
    # I'll use a specific decomposition. Let me define it.
    # 
    # Actually, for simplicity and correctness, let me use the Kronecker product of two rank-7 Strassen decompositions.
    # This gives rank 49, which is valid and better than 64.
    
    # Strassen rank-7 decomposition for 2x2:
    # U2, V2, W2 are 4x7 matrices.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2[i,j] where i is row of A flattened, j is column (M1..M7)
    # A = [[a,b],[c,d]], indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # U2:
    # M1: a+d -> U2[0][0]=1, U2[3][0]=1
    # M2: c+d -> U2[2][1]=1, U2[3][1]=1
    # M3: a(f-h) -> U2[0][2]=1 (for a)
    # M4: d(g-e) -> U2[3][3]=1 (for d)
    # M5: (a+b)h -> U2[0][4]=1, U2[1][4]=1
    # M6: (c-a)(e+f) -> U2[2][5]=1, U2[0][5]=-1
    # M7: (b-d)(g+h) -> U2[1][6]=1, U2[3][6]=-1
    
    U2 = [[0]*7 for _ in range(4)]
    U2[0][0] = 1; U2[3][0] = 1  # M1: a+d
    U2[2][1] = 1; U2[3][1] = 1  # M2: c+d
    U2[0][2] = 1                 # M3: a
    U2[3][3] = 1                 # M4: d
    U2[0][4] = 1; U2[1][4] = 1  # M5: a+b
    U2[2][5] = 1; U2[0][5] = -1 # M6: c-a
    U2[1][6] = 1; U2[3][6] = -1 # M7: b-d
    
    # V2:
    # M1: e+h -> V2[0][0]=1, V2[3][0]=1
    # M2: e -> V2[0][1]=1
    # M3: f-h -> V2[1][2]=1, V2[3][2]=-1
    # M4: g-e -> V2[2][3]=1, V2[0][3]=-1
    # M5: h -> V2[3][4]=1
    # M6: e+f -> V2[0][5]=1, V2[1][5]=1
    # M7: g+h -> V2[2][6]=1, V2[3][6]=1
    
    V2 = [[0]*7 for _ in range(4)]
    V2[0][0] = 1; V2[3][0] = 1  # M1: e+h
    V2[0][1] = 1                 # M2: e
    V2[1][2] = 1; V2[3][2] = -1 # M3: f-h
    V2[2][3] = 1; V2[0][3] = -1 # M4: g-e
    V2[3][4] = 1                 # M5: h
    V2[0][5] = 1; V2[1][5] = 1  # M6: e+f
    V2[2][6] = 1; V2[3][6] = 1  # M7: g+h
    
    # W2:
    # C00 = M1 + M4 - M5 + M7 -> W2[0][0]=1, W2[0][3]=1, W2[0][4]=-1, W2[0][6]=1
    # C01 = M3 + M5 -> W2[1][2]=1, W2[1][4]=1
    # C10 = M2 + M4 -> W2[2][1]=1, W2[2][3]=1
    # C11 = M1 - M2 + M3 + M6 -> W2[3][0]=1, W2[3][1]=-1, W2[3][2]=1, W2[3][5]=1
    
    W2 = [[0]*7 for _ in range(4)]
    W2[0][0] = 1; W2[0][3] = 1; W2[0][4] = -1; W2[0][6] = 1
    W2[1][2] = 1; W2[1][4] = 1
    W2[2][1] = 1; W2[2][3] = 1
    W2[3][0] = 1; W2[3][1] = -1; W2[3][2] = 1; W2[3][5] = 1
    
    # Now construct 4x4 using Kronecker product.
    # The 4x4 tensor is T4 = T2 ⊗ T2.
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49).
    # 
    # Index mapping:
    # For 4x4, A has indices (i,j) where i in 0..3, j in 0..3.
    # We can think of i = 2*i1 + i2, j = 2*j1 + j2, where i1, i2, j1, j2 in 0..1.
    # Similarly for B and C.
    # 
    # U4[(i1,i2,j1,j2), (r1,r2)] = U2[(i1,j1), r1] * U2[(i2,j2), r2]
    # where (i1,i2,j1,j2) is flattened to i1*4 + i2*2 + j1*1 + j2*0? No.
    # 
    # Let me use numpy for the Kronecker product to be safe.
    import numpy as np
    
    U2_arr = np.array(U2, dtype=float)
    V2_arr = np.array(V2, dtype=float)
    W2_arr = np.array(W2, dtype=float)
    
    # The Kronecker product of U2 with U2 gives a 16x49 matrix.
    # But we need to be careful about the index ordering.
    # 
    # A2 is a 2x2 matrix. A4 is a 4x4 matrix.
    # A4 = A2_top_left ⊕ A2_top_right ⊕ A2_bottom_left ⊕ A2_bottom_right, but with the same A2.
    # 
    # Actually, the tensor T4 for 4x4 is not simply T2 ⊗ T2 in the sense that the inputs are independent.
    # The tensor for multiplying A (4x4) by B (4x4) to get C (4x4) is:
    # C[i,j] = sum_{k,l} A[i,k] * B[k,j]
    # 
    # If we write i = 2*i1 + i2, k = 2*k1 + k2, j = 2*j1 + j2, then:
    # C[2*i1+i2, 2*j1+j2] = sum_{k1,k2} A[2*i1+i2, 2*k1+k2] * B[2*k1+k2, 2*j1+j2]
    # 
    # This is not separable in a simple Kronecker product way because A and B are full 4x4 matrices.
    # 
    # However, the tensor T_{n,m,p} for matrix multiplication is not a Kronecker product of smaller tensors in a straightforward way.
    # 
    # So the Kronecker product approach doesn't directly work for reducing rank below n*m*p.
    # 
    # Instead, let me just use the schoolbook decomposition but with a better known rank-49 or lower.
    # 
    # Actually, the minimum known rank for 4x4 is 21 (from the 3x3 rank-19 plus padding, or similar).
    # But I don't have a verified rank-21 for 4x4.
    # 
    # Let me use a safe rank-49 decomposition based on Strassen for 2x2 blocks, but applied correctly.
    # 
    # Actually, a simple approach: use 7 Strassen decompositions for each of the 7 "slices" of the 4x4? No.
    # 
    # Let me just output a valid rank-64 schoolbook decomposition to be safe, but that's score 1.0.
    # 
    # Wait, the current program is score 1.0 with rank 64. I need to do better.
    # 
    # Let me use a rank-49 decomposition. I can construct it by using 7 Strassen decompositions for 2x2, but how?
    # 
    # Actually, I can use the following: 
    # The 4x4 matrix multiplication can be done by 49 multiplications using the following method:
    # Divide A and B into 2x2 blocks. Use Strassen to multiply each pair of blocks, but with sharing.
    # 
    # This is getting complex. Let me use a simple valid decomposition with rank < 64.
    # 
    # I'll use a rank-49 decomposition by padding a 3x3 rank-19 or rank-21 decomposition.
    # 
    # For 3x3, a rank-21 decomposition exists. Let me define one.
    # 
    # From a known source, a rank-21 decomposition for 3x3:
    # 
    # I'll use a specific one. Let me define U3, V3, W3 as 9x21 matrices.
    # 
    # This is complex to implement from scratch. Let me use a different approach.
    # 
    # I'll use the following: 
    # Use 7 Strassen decompositions for 2x2, each handling a subset of the 4x4 multiplication.
    # 
    # Actually, let me just use a rank-49 decomposition by using 49 independent bilinear forms, which is trivially valid but not minimal.
    # 
    # Wait, I can use a simple improvement: use the fact that the diagonal elements can be computed more efficiently.
    # 
    # Let me just output a valid rank-49 decomposition. I'll construct it by using 49 terms, where each term corresponds to a specific (i,j,k) but with some sharing.
    # 
    # Actually, the simplest is to use the schoolbook but with rank 64. That's score 1.0.
    # 
    # I need to do better. Let me use a rank-49 decomposition based on the following idea:
    # Use 7 multiplications for each of the 7 "diagonal" blocks, but this is not standard.
    # 
    # Let me use a known rank-49 decomposition for 4x4.
    # 
    # Actually, I found a simpler approach: 
    # Use the rank-7 Strassen decomposition for 2x2, and apply it to the 4x4 by considering it as a 2x2 block matrix where each block is 2x2.
    # 
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each of these is a sum of two 2x2 matrix products.
    # 
    # We can compute each 2x2 product using Strassen (7 mults), so 8 products * 7 = 56 mults.
    # But we can share some computations.
    # 
    # For example, to compute C11 and C21, both need A11*B11 and A21*B11.
    # 
    # This is getting complex. Let me just use a rank-56 decomposition (8*7), which is better than 64.
    # 
    # Actually, let me use a rank-49 decomposition by being clever.
    # 
    # I'll use the following: 
    # Compute 49 bilinear forms.
    # 
    # Let me just output a valid rank-49 decomposition. I'll use the following construction:
    # 
    # For each (i,j) in 0..3, and each (k) in 0..3, we have a term A[i,k]*B[k,j].
    # That's 16*4 = 64 terms.
    # 
    # To reduce to 49, I need to combine some terms.
    # 
    # This is not straightforward. Let me use a different approach.
    # 
    # I'll use the rank-21 decomposition for 3x3 (which I'll define), and then embed it in 4x4 by adding one row and column.
    # 
    # For 4x4, I can compute the 3x3 submatrix (top-left) using the rank-21 decomposition, and then compute the remaining 7 terms (the last row and last column, excluding the bottom-right) using 7 additional terms, and the bottom-right using 1 term. Total 21 + 7 + 1 = 29? No, that's not right.
    # 
    # Actually, if I have a rank-21 decomposition for 3x3, I can compute C[0:3,0:3] using 21 mults.
    # Then C[0:3,3] requires 3 mults (A[0:3,3]*B[3,0:3] + ... no, C[i,3] = sum_k A[i,k]*B[k,3]).
    # 
    # This is not separable.
    # 
    # Let me just use a rank-49 decomposition. I'll construct it as follows:
    # Use 49 independent terms, where each term is a product of a linear form in A and a linear form in B.
    # 
    # I'll use the following 49 terms:
    # For each (r1, r2) in 0..6 x 0..6, define a term.
    # 
    # Actually, let me use the Kronecker product of U2, V2, W2 in a valid way.
    # 
    # The tensor T4 for 4x4 matrix multiplication can be written as:
    # T4 = sum_{i,j,k} E_{i,k}^A ⊗ E_{k,j}^B ⊗ E_{i,j}^C
    # 
    # This is not a Kronecker product of T2.
    # 
    # I give up on finding a better decomposition quickly. Let me use a rank-64 decomposition but with a different structure to avoid being identical to the seed.
    # 
    # Actually, the seed uses rank 64. I need to do better.
    # 
    # Let me use a rank-56 decomposition (8 Strassen decompositions for 8 block products).
    
    # I'll construct a rank-56 decomposition.
    # 
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Compute each block product using Strassen (7 mults each).
    # 8 block products * 7 = 56 mults.
    # 
    # Let me implement this.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 8 block products:
    # (A11, B11) -> contributes to C11
    # (A12, B21) -> contributes to C11
    # (A11, B12) -> contributes to C12
    # (A12, B22) -> contributes to C12
    # (A21, B11) -> contributes to C21
    # (A22, B21) -> contributes to C21
    # (A21, B12) -> contributes to C22
    # (A22, B22) -> contributes to C22
    
    # For each block product, use Strassen's 7 mults.
    # 
    # Block (A_block, B_block) where A_block is 2x2, B_block is 2x2.
    # A_block indices in A: (base_i + di, base_j + dj)
    # B_block indices in B: (base_k + dk, base_l + dl)
    # C_block indices in C: (base_c_i + dc_i, base_c_j + dc_j)
    
    # Define the 8 blocks:
    # A11: i in 0..1, j in 0..1 -> base_i=0, base_j=0
    # A12: i in 0..1, j in 2..3 -> base_i=0, base_j=2
    # A21: i in 2..3, j in 0..1 -> base_i=2, base_j=0
    # A22: i in 2..3, j in 2..3 -> base_i=2, base_j=2
    # B11: k in 0..1, l in 0..1 -> base_k=0, base_l=0
    # B12: k in 0..1, l in 2..3 -> base_k=0, base_l=2
    # B21: k in 2..3, l in 0..1 -> base_k=2, base_l=0
    # B22: k in 2..3, l in 2..3 -> base_k=2, base_l=2
    # C11: i in 0..1, j in 0..1 -> base_c_i=0, base_c_j=0
    # C12: i in 0..1, j in 2..3 -> base_c_i=0, base_c_j=2
    # C21: i in 2..3, j in 0..1 -> base_c_i=2, base_c_j=0
    # C22: i in 2..3, j in 2..3 -> base_c_i=2, base_c_j=2
    
    blocks = [
        # (A_block, B_block, C_block)
        # A_block: (base_i, base_j), B_block: (base_k, base_l), C_block: (base_c_i, base_c_j)
        ((0, 0), (0, 0), (0, 0)),  # A11 * B11 -> C11
        ((0, 2), (2, 0), (0, 0)),  # A12 * B21 -> C11
        ((0, 0), (0, 2), (0, 2)),  # A11 * B12 -> C12
        ((0, 2), (2, 2), (0, 2)),  # A12 * B22 -> C12
        ((2, 0), (0, 0), (2, 0)),  # A21 * B11 -> C21
        ((2, 2), (2, 0), (2, 0)),  # A22 * B21 -> C21
        ((2, 0), (0, 2), (2, 2)),  # A21 * B12 -> C22
        ((2, 2), (2, 2), (2, 2)),  # A22 * B22 -> C22
    ]
    
    col = 0
    for (ai, aj), (bk, bl), (ci, cj) in blocks:
        # Use Strassen for this block product.
        # U2, V2, W2 are defined above.
        for r in range(7):
            # U contribution
            for di in range(2):
                for dj in range(2):
                    a_idx = (ai + di) * m + (aj + dj)
                    U[a_idx][col] += U2[di * 2 + dj][r]
            # V contribution
            for dk in range(2):
                for dl in range(2):
                    b_idx = (bk + dk) * p + (bl + dl)
                    V[b_idx][col] += V2[dk * 2 + dl][r]
            # W contribution
            for dc_i in range(2):
                for dc_j in range(2):
                    c_idx = (ci + dc_i) * n + (cj + dc_j)
                    W[c_idx][col] += W2[dc_i * 2 + dc_j][r]
            col += 1
    
    # Convert to lists of lists with integer entries
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
