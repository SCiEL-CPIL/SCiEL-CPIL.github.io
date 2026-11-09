"""Program for 4x4 matrix multiplication tensor decomposition."""

def construct():
    # We use a recursive Strassen decomposition.
    # For 4x4 matrices, we can view them as 2x2 blocks of 2x2 matrices.
    # Strassen for 2x2 blocks requires 7 scalar multiplications.
    # Each scalar multiplication is a 2x2 * 2x2 matrix multiplication.
    # We can further decompose these 2x2 multiplications.
    
    # Let's try to build the Strassen decomposition explicitly for 4x4.
    
    # The 4x4 matrices A and B are divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]  where each Ai,j is 2x2
    # B = [[B00, B01], [B10, B11]]
    
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product. We need to decompose each 2x2*2x2 product.
    # A 2x2 * 2x2 product can be done with 8 scalar multiplications (schoolbook).
    # Total rank = 7 * 8 = 56.
    
    # But wait, can we do better? The known best for 4x4 is 49 (by Landsberg).
    # However, implementing 49 is complex. Let's start with Strassen (56) and see if we can improve.
    
    # Actually, let's just implement the standard Strassen-based decomposition which gives rank 56.
    # This is better than 64 (schoolbook).
    
    n, m, p = 4, 4, 4
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block structure:
    # Index mapping for 4x4 matrix:
    # Row index r (0..3), col index c (0..3)
    # Block (i, j) where i = r // 2, j = c // 2
    # Within block: local row lr = r % 2, local col lc = c % 2
    # So global index for U: row = r * m + c = r * 4 + c
    # For V: row = r * p + c = r * 4 + c (since m=p=4)
    # For W: output index. The task says W[k*n+i, r].
    # This is the trace convention. Let's check the seed program.
    # Seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So for multiplication A * B = C:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In the tensor decomposition, the term is U[i*m+j] * V[j*p+k] * W[k*n+i]
    # So W's row index is k*n + i.
    # Here i is row of A (and C), k is col of B (and C).
    
    # Let's define helper to get global index
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    # Strassen blocks:
    # A blocks:
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # For each Strassen term t (0..6), we have a product of two linear combinations.
    # Let's define the coefficients for each term.
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # Term 2: (A10 + A11) * B00
    # Term 3: A00 * (B01 - B11)
    # Term 4: A11 * (B10 - B00)
    # Term 5: (A00 + A01) * B11
    # Term 6: (A10 - A00) * (B00 + B01)
    # Term 7: (A01 - A11) * (B10 + B11)
    
    # For each term, we need to compute the product of two 2x2 block sums.
    # Let's define for each term t:
    # Left side: sum of blocks A_{i,j} with coefficient a_{i,j}
    # Right side: sum of blocks B_{k,l} with coefficient b_{k,l}
    # The product is sum_{i,j,k,l} a_{i,j} b_{k,l} A_{i,j} B_{k,l}
    # But this is not quite right because the blocks are 2x2 matrices.
    # The product of two 2x2 block sums is a 2x2 block matrix where each entry is a 2x2 matrix product.
    
    # Let's think differently.
    # For term t, we have L_t = sum_{(i,j) in support_t} a_{i,j} A_{i,j}  (a 4x4 matrix with only 2x2 blocks non-zero)
    # R_t = sum_{(k,l) in support_t} b_{k,l} B_{k,l}  (a 4x4 matrix with only 2x2 blocks non-zero)
    # Then M_t = L_t * R_t is a 4x4 matrix.
    # But M_t is not a rank-1 term in the tensor decomposition.
    # M_t is a 4x4 matrix product of two 4x4 sparse matrices.
    # We need to decompose L_t * R_t into rank-1 terms.
    
    # L_t has non-zero blocks at positions (i,j) for (i,j) in support_t.
    # R_t has non-zero blocks at positions (k,l) for (k,l) in support_t.
    # The product L_t * R_t has non-zero blocks at positions (i,l) where there exists k such that (i,k) and (k,l) are in the supports.
    # Each non-zero block in L_t * R_t is a 2x2 * 2x2 product.
    
    # Let's enumerate the blocks in each term.
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # L_1 has blocks A00 (at 0,0) and A11 (at 1,1)
    # R_1 has blocks B00 (at 0,0) and B11 (at 1,1)
    # Product blocks:
    # (0,0): A00 * B00
    # (0,1): A00 * B11
    # (1,0): A11 * B00
    # (1,1): A11 * B11
    
    # Each of these is a 2x2 * 2x2 product, which requires 8 scalar multiplications.
    # Total for term 1: 4 * 8 = 32? No, that's not right.
    
    # Wait, L_1 * R_1 is a 4x4 matrix. Its (i,k) block (where i,k are 0..1) is:
    # C_{i,k} = sum_j L_{i,j} R_{j,k}
    # For term 1:
    # C_{0,0} = L_{0,0} R_{0,0} = A00 * B00
    # C_{0,1} = L_{0,0} R_{0,1} + L_{0,1} R_{1,1} = A00 * B11 + 0 = A00 * B11
    # C_{1,0} = L_{1,0} R_{0,0} + L_{1,1} R_{1,0} = 0 + A11 * B00 = A11 * B00
    # C_{1,1} = L_{1,0} R_{0,1} + L_{1,1} R_{1,1} = 0 + A11 * B11 = A11 * B11
    
    # So each C_{i,k} is a 2x2 matrix product of two 2x2 matrices.
    # We need to decompose A00 * B00, A00 * B11, A11 * B00, A11 * B11.
    
    # Each 2x2 * 2x2 product can be decomposed into 8 rank-1 terms.
    # So term 1 contributes 4 * 8 = 32 rank-1 terms? That would give total rank 7 * 32 = 224, which is worse than schoolbook.
    
    # I think I'm misunderstanding the Strassen decomposition in the context of tensor rank.
    
    # The Strassen algorithm computes 7 matrix multiplications of size 2x2 (in the block sense).
    # Each "matrix multiplication" M_t = L_t * R_t is a 4x4 * 4x4 product? No.
    
    # Let's reconsider. In Strassen for 4x4, we treat the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # The 7 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # ...
    # Here, A00, A11, etc. are 2x2 matrices.
    # So M1 is a 2x2 matrix (not 4x4), computed as the product of two 2x2 matrices.
    # But (A00 + A11) is a 2x2 matrix (sum of two 2x2 blocks).
    # So M1 is a 2x2 matrix product.
    
    # Then C00 = M1 + M4 - M5 + M7, etc.
    # Each C_{i,k} is a 2x2 matrix.
    
    # So the total computation is 7 products of 2x2 matrices.
    # Each 2x2 * 2x2 product has tensor rank 8 (schoolbook) or 7 (Strassen for 2x2).
    # If we use schoolbook for each 2x2 product, total rank = 7 * 8 = 56.
    # If we use Strassen for each 2x2 product, total rank = 7 * 7 = 49.
    
    # But the linear combinations (sums/differences) are free (just additions in the U, V, W factors).
    
    # So let's implement rank 56 first (Strassen for 4x4 with schoolbook for 2x2).
    
    # The 7 terms are products of 2x2 matrices.
    # Let's define the 7 terms:
    # T1 = (A00 + A11) * (B00 + B11)
    # T2 = (A10 + A11) * B00
    # T3 = A00 * (B01 - B11)
    # T4 = A11 * (B10 - B00)
    # T5 = (A00 + A01) * B11
    # T6 = (A10 - A00) * (B00 + B01)
    # T7 = (A01 - A11) * (B10 + B11)
    
    # For each term, the left factor is a sum of 2x2 blocks of A, and the right factor is a sum of 2x2 blocks of B.
    # The product is a 2x2 matrix, and its entries contribute to specific C blocks.
    
    # Let's map the contributions:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # Each T_t is a 2x2 matrix. Let's say T_t has entries (t_{0,0}, t_{0,1}, t_{1,0}, t_{1,1}).
    # These entries are linear combinations of products of entries of A and B.
    
    # For the tensor decomposition, we need to express each product of entries as a rank-1 term.
    
    # Let's implement this step by step.
    
    # First, let's define the 2x2 * 2x2 schoolbook decomposition.
    # For matrices X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]],
    # Z = X * Y has:
    # z00 = x00*y00 + x01*y10
    # z01 = x00*y01 + x01*y11
    # z10 = x10*y00 + x11*y10
    # z11 = x10*y01 + x11*y11
    
    # This requires 8 scalar multiplications.
    
    # Now, for each Strassen term T_t, the left factor L_t is a linear combination of A blocks,
    # and the right factor R_t is a linear combination of B blocks.
    
    # Let's define the coefficients for each term.
    
    # We'll use a dictionary to store the coefficients.
    # For term t, left[t] is a dict mapping (i,j) to coefficient, where (i,j) is the block position in A.
    # Similarly for right[t] for B.
    
    left_coeffs = [
        {(0,0): 1, (1,1): 1},  # T1
        {(1,0): 1, (1,1): 1},  # T2
        {(0,0): 1},             # T3
        {(1,1): 1},             # T4
        {(0,0): 1, (0,1): 1},  # T5
        {(1,0): 1, (0,0): -1}, # T6
        {(0,1): 1, (1,1): -1}, # T7
    ]
    
    right_coeffs = [
        {(0,0): 1, (1,1): 1},  # T1
        {(0,0): 1},             # T2
        {(0,1): 1, (1,1): -1}, # T3
        {(1,0): 1, (0,0): -1}, # T4
        {(1,1): 1},             # T5
        {(0,0): 1, (0,1): 1},  # T6
        {(1,0): 1, (1,1): 1},  # T7
    ]
    
    # The contributions of each term to C blocks:
    # C00: T1(+1), T4(+1), T5(-1), T7(+1)
    # C01: T3(+1), T5(+1)
    # C10: T2(+1), T4(+1)
    # C11: T1(+1), T2(-1), T3(+1), T6(+1)
    
    contribs = [
        [(0,0, 1), (1,0, 1), (0,1, -1), (1,1, 1)],  # T1 -> C00(+1), C10(+1), C01(-1)? No.
        # Let me re-check:
        # C00 = T1 + T4 - T5 + T7
        # C01 = T3 + T5
        # C10 = T2 + T4
        # C11 = T1 - T2 + T3 + T6
        # So:
        # T1: C00(+1), C11(+1)
        # T2: C10(+1), C11(-1)
        # T3: C01(+1), C11(+1)
        # T4: C00(+1), C10(+1)
        # T5: C00(-1), C01(+1)
        # T6: C11(+1)
        # T7: C00(+1)
    ]
    
    contribs = [
        [(0,0, 1), (1,1, 1)],  # T1
        [(1,0, 1), (1,1, -1)], # T2
        [(0,1, 1), (1,1, 1)],  # T3
        [(0,0, 1), (1,0, 1)],  # T4
        [(0,0, -1), (0,1, 1)], # T5
        [(1,1, 1)],             # T6
        [(0,0, 1)],             # T7
    ]
    
    # Now, for each term t, we have L_t = sum_{(i,j)} a_{i,j} A_{i,j} and R_t = sum_{(k,l)} b_{k,l} B_{k,l}.
    # L_t is a 4x4 matrix with non-zero 2x2 blocks at positions (i,j).
    # R_t is a 4x4 matrix with non-zero 2x2 blocks at positions (k,l).
    # The product L_t * R_t is a 4x4 matrix.
    # The (i,k) block of L_t * R_t is sum_j L_{i,j} R_{j,k}.
    
    # For term t, the non-zero blocks in L_t are at (i,j) where a_{i,j} != 0.
    # The non-zero blocks in R_t are at (j,k) where b_{j,k} != 0.
    # The product block at (i,k) is sum_j a_{i,j} b_{j,k} A_{i,j} B_{j,k}.
    
    # Each A_{i,j} B_{j,k} is a 2x2 * 2x2 product.
    
    # Let's enumerate all the (i,j,k) triples for each term.
    
    for t in range(7):
        lc = left_coeffs[t]
        rc = right_coeffs[t]
        
        # For each (i,j) in lc and (j,k) in rc:
        for (i, j), a in lc.items():
            for (j2, k), b in rc.items():
                if j == j2:
                    # We have a contribution a * b * A_{i,j} * B_{j,k} to the (i,k) block of L_t * R_t.
                    # A_{i,j} is a 2x2 block of A at row block i, col block j.
                    # B_{j,k} is a 2x2 block of B at row block j, col block k.
                    
                    # The product A_{i,j} * B_{j,k} is a 2x2 matrix.
                    # Let's decompose it using schoolbook.
                    
                    # A_{i,j} has entries A[i*2+r, j*2+c] for r,c in 0..1.
                    # B_{j,k} has entries B[j*2+r, k*2+c] for r,c in 0..1.
                    # The product has entries at C[i*2+r, k*2+c].
                    
                    # Schoolbook for 2x2:
                    # P[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # P[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
                    # P[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    # P[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
                    
                    # Each term is a product of one entry of A and one entry of B.
                    
                    # Let's add 8 rank-1 terms for this 2x2 product.
                    
                    # The coefficient for this contribution is a * b.
                    coeff = a * b
                    
                    # For each of the 8 scalar multiplications:
                    # (ar, ac) in A_{i,j}, (br, bc) in B_{j,k}
                    # contributes to C[i*2+cr, k*2+cc] where cr, cc depend on the schoolbook term.
                    
                    # Schoolbook terms:
                    # 1. A[ar,ac] * B[br,bc] where (ar,ac) and (br,bc) are such that:
                    #    ar == br and ac == bc? No.
                    
                    # Let's list the 8 terms:
                    # 1. A[0,0] * B[0,0] -> C[0,0]
                    # 2. A[0,1] * B[1,0] -> C[0,0]
                    # 3. A[0,0] * B[0,1] -> C[0,1]
                    # 4. A[0,1] * B[1,1] -> C[0,1]
                    # 5. A[1,0] * B[0,0] -> C[1,0]
                    # 6. A[1,1] * B[1,0] -> C[1,0]
                    # 7. A[1,0] * B[0,1] -> C[1,1]
                    # 8. A[1,1] * B[1,1] -> C[1,1]
                    
                    # So for each term, we have (ar, ac, br, bc, cr, cc).
                    terms = [
                        (0, 0, 0, 0, 0, 0),
                        (0, 1, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 1),
                        (0, 1, 1, 1, 0, 1),
                        (1, 0, 0, 0, 1, 0),
                        (1, 1, 1, 0, 1, 0),
                        (1, 0, 0, 1, 1, 1),
                        (1, 1, 1, 1, 1, 1),
                    ]
                    
                    for (ar, ac, br, bc, cr, cc) in terms:
                        # A entry: row = i*2 + ar, col = j*2 + ac
                        a_row = i * 2 + ar
                        a_col = j * 2 + ac
                        # B entry: row = j*2 + br, col = k*2 + bc
                        b_row = j * 2 + br
                        b_col = k * 2 + bc
                        # C entry: row = i*2 + cr, col = k*2 + cc
                        c_row = i * 2 + cr
                        c_col = k * 2 + cc
                        
                        # U index: a_row * m + a_col = a_row * 4 + a_col
                        # V index: b_row * p + b_col = b_row * 4 + b_col
                        # W index: c_col * n + c_row = c_col * 4 + c_row
                        
                        # Find a column for this rank-1 term.
                        # We need to check if this (U_idx, V_idx, W_idx) combination already exists.
                        # If so, we can merge the coefficients.
                        # Otherwise, we add a new column.
                        
                        # For simplicity, let's just add a new column for each term.
                        # This might lead to more columns than necessary, but let's start simple.
                        
                        # Actually, we can merge if the (U_idx, V_idx, W_idx) triple is the same.
                        
                        # Let's use a dictionary to track existing columns.
                        # Key: (u_idx, v_idx, w_idx)
                        # Value: column index
                        # We'll do this in the outer loop.
                        
                        pass  # We'll handle this differently.
    
    # Let's restart with a cleaner approach.
    # We'll build the U, V, W matrices by iterating over all possible rank-1 terms.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Dictionary to merge identical rank-1 terms
    term_map = {}
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal rank
        key = (u_idx, v_idx, w_idx)
        if key in term_map:
            col = term_map[key]
            U[col][u_idx] += coeff
            V[col][v_idx] += coeff
            W[col][w_idx] += coeff
        else:
            term_map[key] = rank
            # Initialize column
            U.append([0] * (n * m))
            V.append([0] * (m * p))
            W.append([0] * (p * n))
            U[rank][u_idx] = coeff
            V[rank][v_idx] = coeff
            W[rank][w_idx] = coeff
            rank += 1
    
    # Now iterate over all Strassen terms and schoolbook subterms.
    for t in range(7):
        lc = left_coeffs[t]
        rc = right_coeffs[t]
        
        for (i, j), a in lc.items():
            for (j2, k), b in rc.items():
                if j == j2:
                    coeff = a * b
                    # Add 8 schoolbook terms for A_{i,j} * B_{j,k}
                    terms = [
                        (0, 0, 0, 0, 0, 0),
                        (0, 1, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 1),
                        (0, 1, 1, 1, 0, 1),
                        (1, 0, 0, 0, 1, 0),
                        (1, 1, 1, 0, 1, 0),
                        (1, 0, 0, 1, 1, 1),
                        (1, 1, 1, 1, 1, 1),
                    ]
                    for (ar, ac, br, bc, cr, cc) in terms:
                        a_row = i * 2 + ar
                        a_col = j * 2 + ac
                        b_row = j * 2 + br
                        b_col = k * 2 + bc
                        c_row = i * 2 + cr
                        c_col = k * 2 + cc
                        
                        u_idx = a_row * m + a_col
                        v_idx = b_row * p + b_col
                        w_idx = c_col * n + c_row
                        
                        add_term(u_idx, v_idx, w_idx, coeff)
    
    # The rank is now the number of columns.
    # Let's return the result.
    return {"U": U, "V": V, "W": W}
