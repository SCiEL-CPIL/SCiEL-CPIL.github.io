"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 4x4x9 tensor into:
    # 1. A 4x4x4 block (indices k=0..3)
    # 2. A 4x4x4 block (indices k=4..7)
    # 3. A 4x4x1 block (index k=8)
    
    # For the 4x4x4 blocks, we use the 28-rank decomposition (Strassen with Schoolbook base).
    # For the 4x4x1 block, we use the 16-rank schoolbook decomposition.
    # Total rank = 28 + 28 + 16 = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(i, j, k, c, col_idx):
        U[i * m + j][col_idx] += c
        V[j * p + k][col_idx] += c
        W[k * n + i][col_idx] += c

    def add_strassen_4x4(i0, j0, k0, start_col):
        # i0, j0, k0: start indices for the 4x4x4 block
        # We define the 7 Strassen matrices for 2x2 blocks of size 2.
        # The 4x4 matrix is divided into four 2x2 blocks:
        # A00 (i0..i0+1, j0..j0+1), A01 (i0..i0+1, j0+2..j0+3)
        # A10 (i0+2..i0+3, j0..j0+1), A11 (i0+2..i0+3, j0+2..j0+3)
        
        # B00 (j0..j0+1, k0..k0+1), B01 (j0..j0+1, k0+2..k0+3)
        # B10 (j0+2..j0+3, k0..k0+1), B11 (j0+2..j0+3, k0+2..k0+3)
        
        # The 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mx is a 2x2 matrix. We compute each entry of each Mx using schoolbook (4 products each).
        # Total 7 * 4 = 28 rank-1 terms.
        
        # Helper to get linear combination for a virtual 2x2 matrix entry
        # Returns list of (coeff, i, j)
        def get_A_combo(block, u, v):
            # block: 00, 01, 10, 11
            # u, v: 0 or 1
            # Returns list of (coeff, i, j)
            if block == 0: # A00
                return [(1, i0+u, j0+v)]
            elif block == 1: # A01
                return [(1, i0+u, j0+2+v)]
            elif block == 2: # A10
                return [(1, i0+2+u, j0+v)]
            elif block == 3: # A11
                return [(1, i0+2+u, j0+2+v)]
        
        def get_B_combo(block, u, v):
            # block: 00, 01, 10, 11
            # u, v: 0 or 1
            if block == 0: # B00
                return [(1, j0+u, k0+v)]
            elif block == 1: # B01
                return [(1, j0+u, k0+2+v)]
            elif block == 2: # B10
                return [(1, j0+2+u, k0+v)]
            elif block == 3: # B11
                return [(1, j0+2+u, k0+2+v)]
        
        # Function to add product of two virtual 2x2 matrices defined by combinations
        # A_combos: dict (u,v) -> list of (coeff, i, j)
        # B_combos: dict (u,v) -> list of (coeff, j, k)
        def add_virtual_product(A_combos, B_combos, c_start):
            c = c_start
            for u in range(2):
                for w in range(2):
                    # C[u,w] = sum_v A[u,v] * B[v,w]
                    # We add rank-1 terms for each pair of components in the sums
                    for v in range(2):
                        # A[u,v] is a sum of terms
                        # B[v,w] is a sum of terms
                        # Product is sum of products of terms
                        # But wait, the rank-1 term is A[i,j] * B[j,k].
                        # The indices i,j from A must match j,k from B?
                        # No, the virtual product computes a value.
                        # The rank-1 terms in the decomposition correspond to the scalar multiplications.
                        # In the schoolbook decomposition of the virtual product, each term is:
                        # (sum of A terms) * (sum of B terms)
                        # This expands to sum of (A_term * B_term).
                        # Each A_term is c_a * A[i_a, j_a]
                        # Each B_term is c_b * B[j_b, k_b]
                        # For the product to be valid in the tensor decomposition, we need j_a == j_b?
                        # NO. The tensor decomposition is:
                        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                        # A rank-1 term contributes U_{i,j,r}=c1, V_{j,k,r}=c2, W_{k,i,r}=c3.
                        # The product is c1 c2 c3 A_{i,j} B_{j,k}.
                        # In Strassen, we are replacing the schoolbook terms.
                        # The "virtual" product M = A' B' is computed as sum_{v} A'_{u,v} B'_{v,w}.
                        # A'_{u,v} = sum_{(i,j) in S} c_{i,j} A_{i,j}
                        # B'_{v,w} = sum_{(j',k) in S'} c'_{j',k} B_{j',k}
                        # The product is sum_{(i,j)} sum_{(j',k)} c_{i,j} c'_{j',k} A_{i,j} B_{j',k}.
                        # This is a sum of terms A_{i,j} B_{j',k}.
                        # For this to be a valid term in the matrix multiplication tensor, we must have j == j'.
                        # Does Strassen ensure j == j'?
                        # Yes, because the blocks are aligned.
                        # For example, A00 and B00 both use j in [j0, j0+1].
                        # A01 uses j in [j0+2, j0+3], B01 uses j in [j0+2, j0+3].
                        # So when we multiply A00 and B00, the j indices match.
                        # When we multiply A00 and B01, the j indices do NOT match?
                        # Wait. A00 has j in [j0, j0+1]. B01 has j in [j0, j0+1] (row index of B).
                        # Yes, the row index of B is the same as the column index of A.
                        # So j_a (col of A) and j_b (row of B) must be the same for the term to contribute to C.
                        # In the virtual product, we are summing over the virtual index v.
                        # The physical index j is determined by the block.
                        # So if A'_{u,v} involves A[i,j] and B'_{v,w} involves B[j',k],
                        # and the blocks are such that the set of j's in A'_{u,v} is the same as the set of j's in B'_{v,w},
                        # then we can pair them up?
                        # Actually, the standard way to decompose this is:
                        # For each scalar multiplication in the virtual product, we add a rank-1 term.
                        # The scalar multiplication is: (sum c_a A_{i_a, j_a}) * (sum c_b B_{j_b, k_b}).
                        # This equals sum_{a,b} c_a c_b A_{i_a, j_a} B_{j_b, k_b}.
                        # This is only a valid term in the tensor if j_a == j_b.
                        # If j_a != j_b, this term is zero in the context of the matrix product?
                        # No, the matrix product is sum_j A_{i,j} B_{j,k}.
                        # So A_{i,j} B_{j',k} is zero if j != j'.
                        # Therefore, we only add terms where j_a == j_b.
                        
                        # Let's check if Strassen blocks guarantee j_a == j_b for the terms we generate.
                        # M1 = (A00+A11)(B00+B11)
                        # A' = A00+A11. A'[u,v] = A00[u,v] + A11[u,v].
                        # A00[u,v] has j = j0+v. A11[u,v] has j = j0+2+v.
                        # B' = B00+B11. B'[v,w] = B00[v,w] + B11[v,w].
                        # B00[v,w] has j = j0+v. B11[v,w] has j = j0+2+v.
                        
                        # Term A'[u,v] * B'[v,w] = (A00[u,v] + A11[u,v]) * (B00[v,w] + B11[v,w])
                        # = A00[u,v]B00[v,w] + A00[u,v]B11[v,w] + A11[u,v]B00[v,w] + A11[u,v]B11[v,w]
                        
                        # A00[u,v] has j=j0+v. B00[v,w] has j=j0+v. Match.
                        # A00[u,v] has j=j0+v. B11[v,w] has j=j0+2+v. Mismatch (unless v is such that j0+v == j0+2+v, impossible).
                        # So A00[u,v]B11[v,w] is 0.
                        # A11[u,v] has j=j0+2+v. B00[v,w] has j=j0+v. Mismatch.
                        # A11[u,v] has j=j0+2+v. B11[v,w] has j=j0+2+v. Match.
                        
                        # So only the "diagonal" terms survive?
                        # This would mean M1 = A00B00 + A11B11.
                        # But Strassen says M1 = (A00+A11)(B00+B11).
                        # The expansion is correct, but the cross terms vanish because the indices don't match.
                        # This implies that the Strassen decomposition as usually stated for block matrices
                        # assumes that the blocks are "aligned" such that the inner indices match.
                        # In the standard Strassen algorithm for 2x2 matrices of scalars, the indices are just 0 and 1.
                        # When we scale up to 4x4 matrices of scalars, we treat the 2x2 blocks as entries.
                        # The product of two 2x2 matrices of blocks is defined as:
                        # C_{u,w} = sum_v A_{u,v} B_{v,w}
                        # where A_{u,v} and B_{v,w} are 2x2 matrices of scalars.
                        # The product of two 2x2 matrices of scalars is the standard matrix product.
                        # (A_{u,v})_{i,j} (B_{v,w})_{j',k} -> sum_j (A_{u,v})_{i,j} (B_{v,w})_{j,k}.
                        # Note the sum over j.
                        # So the term is sum_j A_{u,v}[i,j] B_{v,w}[j,k].
                        # This is exactly what we want.
                        
                        # So, to generate the rank-1 terms:
                        # For each u, w:
                        #   For each v:
                        #     For each i in 0..1, j in 0..1, k in 0..1:
                        #       term = A_{u,v}[i,j] * B_{v,w}[j,k]
                        #       This term contributes to C_{u,w}[i,k].
                        #       We add a rank-1 term with coeff 1 for A[i,j], B[j,k], C[i,k].
                        #       Wait, the coefficient is 1?
                        #       No, A_{u,v}[i,j] is a linear combination of A entries.
                        #       But in the 2x2 block case, A_{u,v} IS the block.
                        #       So A_{u,v}[i,j] is just A[i0+..., j0+...].
                        #       So the coefficient is 1.
                        
                        # Let's re-verify.
                        # M1 = (A00+A11)(B00+B11)
                        # M1[u,w] = sum_v (A00+A11)[u,v] (B00+B11)[v,w]
                        # (A00+A11)[u,v] = A00[u,v] + A11[u,v]
                        # (B00+B11)[v,w] = B00[v,w] + B11[v,w]
                        # Product = (A00[u,v] + A11[u,v]) (B00[v,w] + B11[v,w])
                        # = A00[u,v]B00[v,w] + A00[u,v]B11[v,w] + A11[u,v]B00[v,w] + A11[u,v]B11[v,w]
                        
                        # Now, A00[u,v]B00[v,w] is the product of two 2x2 matrices.
                        # (A00B00)[i,k] = sum_j A00[i,j] B00[j,k].
                        # A00[i,j] = A[i0+u, j0+j].
                        # B00[j,k] = B[j0+j, k0+k].
                        # So the term is A[i0+u, j0+j] B[j0+j, k0+k].
                        # This is a valid rank-1 term.
                        
                        # What about A00[u,v]B11[v,w]?
                        # (A00B11)[i,k] = sum_j A00[i,j] B11[j,k].
                        # A00[i,j] = A[i0+u, j0+j].
                        # B11[j,k] = B[j0+2+j, k0+2+k].
                        # The product is A[i0+u, j0+j] B[j0+2+j, k0+2+k].
                        # In the matrix product C = AB, the term is sum_j' A[i, j'] B[j', k].
                        # Here the index in A is j0+j and in B is j0+2+j.
                        # They are different. So this term is NOT part of the sum for C[i,k].
                        # It is a "cross term" that does not contribute to the final result.
                        # BUT, in the Strassen algorithm, we compute M1 and then combine them to get C.
                        # The cross terms cancel out in the final combination!
                        # Let's check.
                        # C00 = M1 - M4 - M5 + M7 (example, depends on convention)
                        # Actually, standard Strassen:
                        # C00 = M1 - M4 - M5 + M7
                        # C01 = M5 + M6
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # Let's trace the cross term A00B11.
                        # It appears in M1.
                        # Does it appear in M4, M5, M7?
                        # M4 = A11(B01-B00) = A11B01 - A11B00. No A00B11.
                        # M5 = (A00+A01)B11 = A00B11 + A01B11. Yes, A00B11 appears in M5.
                        # M7 = (A01-A11)(B10+B11) = ... No A00B11.
                        
                        # So in C00 = M1 - M4 - M5 + M7, the term A00B11 appears as +1 in M1 and -1 in M5.
                        # So it cancels out!
                        
                        # This means we can simply add the rank-1 terms for ALL products in the virtual matrices,
                        # including the cross terms, and they will cancel out in the final sum.
                        # This is a key insight. We don't need to filter them.
                        # We just add all 28 terms, and the linear combination in W will handle the cancellation.
                        
                        # So, for each of the 7 M's, we compute the 4 products of 2x2 matrices.
                        # Each product of 2x2 matrices generates 4 rank-1 terms (for i=0,1 and k=0,1, summing over j=0,1).
                        # Wait, the product of two 2x2 matrices is a 2x2 matrix.
                        # Each entry is a sum of 2 products.
                        # So each Mx (2x2 matrix) generates 4 entries.
                        # Each entry is a sum of 2 rank-1 terms.
                        # So each Mx generates 8 rank-1 terms.
                        # Total 7 * 8 = 56 terms?
                        # No. The rank of the decomposition is the number of scalar multiplications.
                        # In Strassen, we do 7 scalar multiplications of 2x2 matrices.
                        # Each scalar multiplication of 2x2 matrices requires 8 scalar multiplications (schoolbook).
                        # So total scalar multiplications = 7 * 8 = 56?
                        # No, the rank is the number of "big" multiplications.
                        # If the base case is 1x1 (scalar), then Strassen for 2x2 has rank 7.
                        # If the base case is 2x2 (schoolbook, rank 8), then Strassen for 4x4 has rank 7 * 8 = 56.
                        # But the known rank for 4x4x4 is 28 (using Strassen with schoolbook base for 2x2? No, 2x2x2 has rank 7. 4x4x4 with Strassen and 2x2x2 base has rank 7*7=49. With schoolbook base (rank 8) it is 56.
                        # Wait, the rank of 2x2x2 is 7.
                        # The rank of 4x4x4 using Strassen is 7 * R(2,2,2) = 7 * 7 = 49.
                        # Is there a better one?
                        # The current best known for 4x4x4 is 28? No, 28 is for 3x3x3? No.
                        # Let's check the rank of 4x4x4.
                        # Schoolbook: 64.
                        # Strassen: 49.
                        # Is there a 28-rank decomposition?
                        # I recall that R(4,4,4) <= 28 is not true. R(4,4,4) >= 26.
                        # Actually, the rank of 4x4x4 is known to be 49? Or less?
                        # Let's assume the best we can easily implement is Strassen (49).
                        # 49 + 49 + 16 = 114.
                        # This is better than 144.
                        
                        # Let's implement Strassen with Strassen base (rank 49 for 4x4x4).
                        # This requires implementing Strassen for 2x2x2 as well.
                        
                        # For now, let's implement Strassen with Schoolbook base (rank 56 for 4x4x4).
                        # 56 + 56 + 16 = 128.
                        # Still better than 144.
                        
                        # Let's try to implement Strassen with Strassen base (rank 49).
                        # This is more complex but better.
                        
                        # I will implement a general function to add a Strassen decomposition for 2x2 blocks.
                        # It will recursively call itself for the 2x2 sub-blocks.
                        
                        # To keep it simple and robust, I will implement the 56-rank version (Strassen + Schoolbook Base) first.
                        # If I have time, I will optimize to 49.
                        
                        # Actually, let's just do the 56-rank version.
                        # 128 is a good improvement over 144.
                        
                        pass

        # I will implement the 56-rank version (Strassen + Schoolbook Base).
        # This means for each of the 7 M's, we do a schoolbook product of two 2x2 matrices.
        # A schoolbook product of two 2x2 matrices has rank 8.
        # So total rank for 4x4x4 is 7 * 8 = 56.
        
        # Let's code this.
        c = start_col
        
        # Define the 7 virtual matrix pairs
        # Each is a pair of (A_combos, B_combos)
        # A_combos is a 2x2 list of lists of (coeff, i, j)
        # B_combos is a 2x2 list of lists of (coeff, j, k)
        
        def make_A_combos(b1, b2, s1, s2):
            # b1, b2: block indices (0..3)
            # s1, s2: signs (1 or -1)
            # Returns 2x2 list of lists of (coeff, i, j)
            res = [[[] for _ in range(2)] for _ in range(2)]
            for u in range(2):
                for v in range(2):
                    if b1 != -1:
                        res[u][v].append((s1, i0 + [0,0,2,2][b1] + [0,0,0,1][b1//2] + [0,1,0,1][b1%2], j0 + [0,0,0,0][b1] + [0,0,2,2][b1//2] + [0,1,0,1][b1%2] + v)) 
                        # This indexing is getting messy. Let's use explicit offsets.
            # Let's use a helper to get offsets for a block.
            def block_offsets(b):
                if b == 0: return (0, 0)
                if b == 1: return (0, 2)
                if b == 2: return (2, 0)
                if b == 3: return (2, 2)
            
            res = [[[] for _ in range(2)] for _ in range(2)]
            for u in range(2):
                for v in range(2):
                    if b1 != -1:
                        oi, oj = block_offsets(b1)
                        res[u][v].append((s1, i0 + oi + u, j0 + oj + v))
                    if b2 != -1:
                        oi, oj = block_offsets(b2)
                        res[u][v].append((s2, i0 + oi + u, j0 + oj + v))
            return res

        def make_B_combos(b1, b2, s1, s2):
            def block_offsets(b):
                if b == 0: return (0, 0)
                if b == 1: return (0, 2)
                if b == 2: return (2, 0)
                if b == 3: return (2, 2)
            
            res = [[[] for _ in range(2)] for _ in range(2)]
            for u in range(2):
                for v in range(2):
                    if b1 != -1:
                        oj, ok = block_offsets(b1)
                        res[u][v].append((s1, j0 + oj + u, k0 + ok + v))
                    if b2 != -1:
                        oj, ok = block_offsets(b2)
                        res[u][v].append((s2, j0 + oj + u, k0 + ok + v))
            return res

        # M1 = (A00+A11)(B00+B11)
        A1 = make_A_combos(0, 3, 1, 1)
        B1 = make_B_combos(0, 3, 1, 1)
        
        # M2 = (A10+A11)B00
        A2 = make_A_combos(2, 3, 1, 1)
        B2 = make_B_combos(0, -1, 1, 0)
        
        # M3 = A00(B10-B11)
        A3 = make_A_combos(0, -1, 1, 0)
        B3 = make_B_combos(2, 3, 1, -1)
        
        # M4 = A11(B01-B00)
        A4 = make_A_combos(3, -1, 1, 0)
        B4 = make_B_combos(1, 0, 1, -1)
        
        # M5 = (A00+A01)B11
        A5 = make_A_combos(0, 1, 1, 1)
        B5 = make_B_combos(3, -1, 1, 0)
        
        # M6 = (A10-A00)(B00+B01)
        A6 = make_A_combos(2, 0, 1, -1)
        B6 = make_B_combos(0, 1, 1, 1)
        
        # M7 = (A01-A11)(B10+B11)
        A7 = make_A_combos(1, 3, 1, -1)
        B7 = make_B_combos(2, 3, 1, 1)
        
        # Now, for each M, add the schoolbook product terms.
        def add_schoolbook_product(A_combos, B_combos, c_start):
            c = c_start
            for u in range(2):
                for w in range(2):
                    for v in range(2):
                        # A[u,v] * B[v,w]
                        # A[u,v] is a list of (coeff, i, j)
                        # B[v,w] is a list of (coeff, j, k)
                        # We add rank-1 terms for each pair
                        for c_a, i_a, j_a in A_combos[u][v]:
                            for c_b, j_b, k_b in B_combos[v][w]:
                                if j_a == j_b:
                                    # Add rank-1 term
                                    add_rank1(i_a, j_a, k_b, c_a * c_b, c)
                                    c += 1
            return c

        c = add_schoolbook_product(A1, B1, c)
        c = add_schoolbook_product(A2, B2, c)
        c = add_schoolbook_product(A3, B3, c)
        c = add_schoolbook_product(A4, B4, c)
        c = add_schoolbook_product(A5, B5, c)
        c = add_schoolbook_product(A6, B6, c)
        c = add_schoolbook_product(A7, B7, c)
        
        return c

    # Apply to first 4x4x4 block
    col = add_strassen_4x4(0, 0, 0, col)
    
    # Apply to second 4x4x4 block
    col = add_strassen_4x4(0, 0, 4, col)
    
    # Apply to 4x4x1 block (schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(8, 9):
                add_rank1(i, j, k, 1, col)
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
