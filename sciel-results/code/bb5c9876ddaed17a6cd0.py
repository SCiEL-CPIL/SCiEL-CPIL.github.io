"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n×m, B is m×p, C is n×p
    
    # We'll build the decomposition using block structure
    # Partition A's rows: [1, 2], A's cols: [2, 3]
    # Partition B's rows: [2, 3] (matching A's cols), B's cols: [3, 4]
    # This gives C blocks: C11 (1×3), C12 (1×4), C21 (2×3), C22 (2×4)
    
    # For each block multiplication, we use either schoolbook or Strassen
    # Let's collect all terms
    
    U_terms = []  # list of (row, col_index, value)
    V_terms = []
    W_terms = []
    
    col = 0
    
    # Block A11 (1×2) * B11 (2×3) -> C11 part (1×3)
    # Schoolbook: 1*2*3 = 6 terms
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_terms.append((0*2 + j, col, 1))  # A row 0, col j
                V_terms.append((0*3 + j, col, 1))  # B row j (in first 2), col k
                W_terms.append((k*1 + 0, col, 1))  # C row k (in first 3), col 0
                col += 1
    
    # Block A12 (1×3) * B12 (2×4) -> C12 part (1×4)
    # Wait, this isn't right. Let me reconsider.
    
    # Actually, C = A @ B where A is 3×5, B is 5×7
    # C[i,k] = sum_j A[i,j] * B[j,k] for j=0..4
    
    # Let me just use a smarter decomposition directly.
    # I'll use the known Strassen-based approach for the 2×2 sub-blocks.
    
    # Clear and restart with a cleaner approach
    U_terms = []
    V_terms = []
    W_terms = []
    col = 0
    
    # I'll partition the computation differently.
    # Use the fact that we can write the multiplication using fewer products
    # by combining terms.
    
    # For now, let's use a block approach:
    # Split middle dimension 5 into 2+3
    # A = [A0 (3×2) | A1 (3×3)]
    # B = [B0 (2×7); B1 (3×7)]
    # C = A0@B0 + A1@B1
    
    # A0@B0: 3×2 × 2×7, schoolbook = 42
    # A1@B1: 3×3 × 3×7, use Strassen: split 3×3 into (1×1, 1×2, 2×1, 2×2)
    #   and 3×7 into (1×3, 1×4, 2×3, 2×4)
    #   Strassen needs 7 products instead of 9 for the 2×2 case, but our blocks are irregular
    
    # Let me just do schoolbook for now but try to find savings.
    # Actually, let me try a completely different approach: 
    # Use the rank structure directly.
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,7)
    # We want to write T as sum of rank-1 tensors.
    
    # A rank-1 tensor is U_a * V_b * W_c where:
    # U_a[i*m+j] is nonzero, V_b[j*p+k] is nonzero, W_c[k*n+i] is nonzero
    # and the product gives T[i,j,k] = 1
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # To reduce rank, we need to find linear dependencies or use Strassen identities.
    
    # Strassen for 2×2: M11 = (A11+A22)(B11+B22), etc.
    # This replaces 4 products with 7.
    
    # Let me apply Strassen to 2×2×2 sub-blocks where possible.
    
    # Identify 2×2×2 sub-blocks in our 3×5×7 tensor:
    # We can fit at most floor(3/2)=1 in i, floor(5/2)=2 in j, floor(7/2)=3 in k
    # So 1*2*3 = 6 disjoint 2×2×2 blocks
    
    # For each 2×2×2 block, Strassen saves 4-7 = -3, so saves 3 per block
    # Total savings: 6 * 3 = 18, giving rank 105 - 18 = 87? That seems too good.
    
    # Wait, Strassen uses 7 products for 2×2×2 instead of 8, saving 1 per block.
    # So 6 blocks save 6, giving 99. That matches the inspiration!
    
    # Let me implement this: find 6 disjoint 2×2×2 sub-blocks and apply Strassen to each.
    
    # The 2×2×2 blocks are at positions:
    # i: 0 (since 3//2 = 1 block starting at 0)
    # j: 0, 2 (since 5//2 = 2 blocks starting at 0, 2)
    # k: 0, 2, 4 (since 7//2 = 3 blocks starting at 0, 2, 4)
    
    # So blocks at (i0, j0, k0) where i0=0, j0 in {0,2}, k0 in {0,2,4}
    
    # For each block, the 8 schoolbook terms become 7 Strassen terms.
    
    # Strassen for 2×2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In our tensor notation, for a 2×2×2 block at (i0,j0,k0):
    # The 8 schoolbook terms are (i0+a, j0+b, k0+c) for a,b,c in {0,1}
    # Each term has U[(i0+a)*5+(j0+b)], V[(j0+b)*7+(k0+c)], W[(k0+c)*3+(i0+a)]
    
    # For Strassen, we need to express the 7 products in terms of these.
    
    # Let me define the 7 Strassen products for block (i0,j0,k0):
    # Each product M_t corresponds to a combination of U, V, W vectors.
    
    # M1 = (A11+A22)(B11+B22): This involves U indices for (i0,j0) and (i0+1,j0+1),
    #    V indices for (j0,k0) and (j0+1,k0+1), W indices for (k0,i0) and (k0+1,i0+1)
    #    But Strassen products are not simply rank-1 in the original sense; they involve sums.
    
    # Actually, each Strassen "product" M_t is itself a bilinear form that can be 
    # decomposed into rank-1 terms. For M1 = (A11+A22)(B11+B22), this is:
    # (A11*B11 + A11*B22 + A22*B11 + A22*B22)
    # which is 4 rank-1 terms. So M1 alone costs 4, not 1!
    
    # I think I'm confusing the level of decomposition. Let me reconsider.
    
    # In the tensor decomposition context, each "term" in the sum is a rank-1 tensor.
    # Strassen reduces the number of scalar multiplications, but in our setting,
    # each scalar multiplication already corresponds to one rank-1 tensor term.
    
    # So Strassen for 2×2×2 uses 7 rank-1 terms instead of 8, saving 1 per block.
    
    # Let me look up how Strassen translates to tensor decomposition terms.
    
    # For 2×2 matrix multiplication, the 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_t is a product of a linear combination of A entries and a linear 
    # combination of B entries. In tensor decomposition, each such product 
    # corresponds to one rank-1 term where:
    # U has the coefficients for the A linear combination
    # V has the coefficients for the B linear combination
    # W has the coefficients for the C linear combination (how this M_t contributes to C entries)
    
    # So yes, 7 terms per 2×2×2 block instead of 8.
    
    # Let me implement this properly.
    
    U_dict = {}  # (row, col) -> value
    V_dict = {}
    W_dict = {}
    
    col = 0
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col
        U_dict[(u_row, col)] = U_dict.get((u_row, col), 0) + u_val
        V_dict[(v_row, col)] = V_dict.get((v_row, col), 0) + v_val
        W_dict[(w_row, col)] = W_dict.get((w_row, col), 0) + w_val
        col += 1
    
    # Process each 2×2×2 block with Strassen
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                # 7 Strassen terms
                # M1 = (A11+A22)(B11+B22)
                # A11 is at (i0, j0), A22 at (i0+1, j0+1)
                # B11 is at (j0, k0), B22 at (j0+1, k0+1)
                # C11 gets +M1, C12 gets +M1, C21 gets +M1, C22 gets +M1 (with signs)
                
                # M1: U has 1 at (i0,j0) and 1 at (i0+1,j0+1)
                #      V has 1 at (j0,k0) and 1 at (j0+1,k0+1)
                #      W has contributions to C11, C12, C21, C22
                # C11 at (k0, i0) gets +1, C12 at (k0+1, i0) gets +1
                # C21 at (k0, i0+1) gets +1, C22 at (k0+1, i0+1) gets +1
                
                # But W is a single vector for this term. The W vector has:
                # W[(k0)*3 + i0] = 1 (C11)
                # W[(k0+1)*3 + i0] = 1 (C12)  
                # W[(k0)*3 + (i0+1)] = 1 (C21)
                # W[(k0+1)*3 + (i0+1)] = 1 (C22)
                
                # Wait, but Strassen M1 contributes to all four C blocks with specific signs.
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # So M1 contributes +1 to C11 and +1 to C22, 0 to C12 and C21.
                # M2 contributes 0 to C11, 0 to C12, +1 to C21, -1 to C22
                # M3 contributes 0 to C11, +1 to C12, 0 to C21, +1 to C22
                # M4 contributes +1 to C11, 0 to C12, +1 to C21, 0 to C22
                # M5 contributes -1 to C11, +1 to C12, 0 to C21, 0 to C22
                # M6 contributes 0 to C11, 0 to C12, 0 to C21, +1 to C22
                # M7 contributes +1 to C11, 0 to C12, 0 to C21, 0 to C22
                
                # Let me define each Strassen term:
                
                # M1 = (A11+A22)(B11+B22)
                add_term(i0*5+j0, j0*7+k0, k0*3+i0, 1, 1, 1)  # A11*B11 -> C11
                add_term(i0*5+j0, j0*7+k0, (k0+1)*3+i0, 1, 1, 1)  # A11*B11 -> C12? No...
                
                # I'm getting confused. Let me think about this differently.
                
                # Each rank-1 term in the decomposition is:
                # T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # For a Strassen term M_t, it's a product of:
                # (sum of some A entries with coefficients) * (sum of some B entries with coefficients)
                # and it contributes to C entries with certain coefficients.
                
                # So for M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1]):
                # This expands to:
                # A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
                # Each of these 4 products contributes to specific C entries.
                
                # But in the tensor decomposition, M1 is ONE term, not 4. The linearity is handled
                # by the U, V, W vectors having multiple nonzero entries.
                
                # U vector for M1: U[i0*5+j0] = 1, U[(i0+1)*5+(j0+1)] = 1
                # V vector for M1: V[j0*7+k0] = 1, V[(j0+1)*7+(k0+1)] = 1
                # W vector for M1: W[k0*3+i0] = 1 (for C11), W[(k0+1)*3+(i0+1)] = 1 (for C22)
                #   (M1 contributes +1 to C11 and +1 to C22)
                
                # Let me redo this properly.
                
                # Clear the dicts and restart
                U_dict = {}
                V_dict = {}
                W_dict = {}
                col = 0
                
                def add_strassen_term(u_entries, v_entries, w_entries):
                    nonlocal col
                    for row, val in u_entries:
                        U_dict[(row, col)] = U_dict.get((row, col), 0) + val
                    for row, val in v_entries:
                        V_dict[(row, col)] = V_dict.get((row, col), 0) + val
                    for row, val in w_entries:
                        W_dict[(row, col)] = W_dict.get((row, col), 0) + val
                    col += 1
                
                # M1 = (A11+A22)(B11+B22)
                # U: A11 at (i0,j0), A22 at (i0+1,j0+1)
                # V: B11 at (j0,k0), B22 at (j0+1,k0+1)
                # W: C11 at (k0,i0) with +1, C22 at (k0+1,i0+1) with +1
                add_strassen_term(
                    [(i0*5+j0, 1), ((i0+1)*5+(j0+1), 1)],
                    [(j0*7+k0, 1), ((j0+1)*7+(k0+1), 1)],
                    [(k0*3+i0, 1), ((k0+1)*3+(i0+1), 1)]
                )
                
                # M2 = (A21+A22)B11
                # U: A21 at (i0+1,j0), A22 at (i0+1,j0+1)
                # V: B11 at (j0,k0)
                # W: C21 at (k0,i0+1) with +1, C22 at (k0+1,i0+1) with -1
                add_strassen_term(
                    [((i0+1)*5+j0, 1), ((i0+1)*5+(j0+1), 1)],
                    [(j0*7+k0, 1)],
                    [(k0*3+(i0+1), 1), ((k0+1)*3+(i0+1), -1)]
                )
                
                # M3 = A11(B21-B22)
                # U: A11 at (i0,j0)
                # V: B21 at (j0+1,k0) with +1, B22 at (j0+1,k0+1) with -1
                # W: C12 at (k0+1,i0) with +1, C22 at (k0+1,i0+1) with +1
                add_strassen_term(
                    [(i0*5+j0, 1)],
                    [((j0+1)*7+k0, 1), ((j0+1)*7+(k0+1), -1)],
                    [((k0+1)*3+i0, 1), ((k0+1)*3+(i0+1), 1)]
                )
                
                # M4 = A22(B12-B11)
                # U: A22 at (i0+1,j0+1)
                # V: B12 at (j0,k0+1) with +1, B11 at (j0,k0) with -1
                # W: C11 at (k0,i0) with +1, C21 at (k0,i0+1) with +1
                add_strassen_term(
                    [((i0+1)*5+(j0+1), 1)],
                    [(j0*7+(k0+1), 1), (j0*7+k0, -1)],
                    [(k0*3+i0, 1), (k0*3+(i0+1), 1)]
                )
                
                # M5 = (A11+A12)B22
                # U: A11 at (i0,j0), A12 at (i0,j0+1)
                # V: B22 at (j0+1,k0+1)
                # W: C11 at (k0,i0) with -1, C12 at (k0+1,i0) with +1
                add_strassen_term(
                    [(i0*5+j0, 1), (i0*5+(j0+1), 1)],
                    [((j0+1)*7+(k0+1), 1)],
                    [(k0*3+i0, -1), ((k0+1)*3+i0, 1)]
                )
                
                # M6 = (A21-A11)(B11+B12)
                # U: A21 at (i0+1,j0) with +1, A11 at (i0,j0) with -1
                # V: B11 at (j0,k0), B12 at (j0,k0+1)
                # W: C22 at (k0+1,i0+1) with +1
                add_strassen_term(
                    [((i0+1)*5+j0, 1), (i0*5+j0, -1)],
                    [(j0*7+k0, 1), (j0*7+(k0+1), 1)],
                    [((k0+1)*3+(i0+1), 1)]
                )
                
                # M7 = (A12-A22)(B21+B22)
                # U: A12 at (i0,j0+1) with +1, A22 at (i0+1,j0+1) with -1
                # V: B21 at (j0+1,k0), B22 at (j0+1,k0+1)
                # W: C11 at (k0,i0) with +1
                add_strassen_term(
                    [(i0*5+(j0+1), 1), ((i0+1)*5+(j0+1), -1)],
                    [((j0+1)*7+k0, 1), ((j0+1)*7+(k0+1), 1)],
                    [(k0*3+i0, 1)]
                )
    
    # Now handle the remaining terms not covered by the 2×2×2 blocks
    # The blocks cover:
    # i: 0,1 (from i0=0)
    # j: 0,1,2,3 (from j0=0,2)
    # k: 0,1,2,3,4,5 (from k0=0,2,4)
    
    # Remaining i: 2
    # Remaining j: 4
    # Remaining k: 6
    
    # Terms to add:
    # For i=2, j in 0..4, k in 0..6: 5*7 = 35 terms
    # For i in 0..1, j=4, k in 0..6: 2*7 = 14 terms
    # For i in 0..1, j in 0..3, k=6: 2*4 = 8 terms
    # Total remaining: 35 + 14 + 8 = 57
    
    # But wait, some of these might overlap with what's already covered.
    # The Strassen blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    # So remaining:
    # i=2, any j, any k: 5*7 = 35
    # i in {0,1}, j=4, any k: 2*7 = 14
    # i in {0,1}, j in {0,1,2,3}, k=6: 2*4 = 8
    # Total: 57
    
    # Add these with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                covered = False
                for i0 in [0]:
                    for j0 in [0, 2]:
                        for k0 in [0, 2, 4]:
                            if (i0 <= i <= i0+1) and (j0 <= j <= j0+1) and (k0 <= k <= k0+1):
                                covered = True
                                break
                        if covered:
                            break
                    if covered:
                        break
                if not covered:
                    add_strassen_term(
                        [(i*5+j, 1)],
                        [(j*7+k, 1)],
                        [(k*3+i, 1)]
                    )
    
    # Convert dicts to lists
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, c), val in U_dict.items():
        U[row][c] = val
    for (row, c), val in V_dict.items():
        V[row][c] = val
    for (row, c), val in W_dict.items():
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

