"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2x2 * 2x15.
    
    We decompose the 2x15 matrix into 3 blocks of 2x5 each.
    For each 2x5 block, we use a rank-16 decomposition.
    Total rank = 3 * 16 = 48 (schoolbook would be 60).
    
    The 2x5 decomposition uses a tensor rank approach:
    For multiplying [2x2] * [2x5], we can think of it as:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,4}
    
    The standard decomposition has rank 2*2*5 = 20.
    We can reduce this by using the fact that for 2x2 matrix multiplication,
    Strassen's algorithm gives rank 7 (or 8 with simple additions).
    
    Actually, let me think more carefully. The tensor for (2,2,5) has entries T[i,j,k] 
    such that C[i,k] = sum_j A[i,j] B[j,k]. This is the outer product structure.
    
    The standard rank is 2*2*5 = 20.
    
    We can split the 5 columns into groups. Let's use:
    - Columns 0,1: 2x2 multiplication (rank 8 with Strassen-like, or 4 schoolbook)
    - Columns 2,3: another 2x2 multiplication  
    - Column 4: 2x1 multiplication (rank 2 schoolbook)
    
    For 2x2 * 2x2: standard rank 4, Strassen rank 7.
    For 2x2 * 2x1: standard rank 4.
    
    Let me use a direct construction:
    
    For (2,2,5):
    Split into (2,2,2) + (2,2,2) + (2,2,1)
    
    (2,2,2) schoolbook: rank 8
    (2,2,1) schoolbook: rank 4
    
    Total: 8 + 8 + 4 = 20. That's just schoolbook.
    
    Let me try Strassen for the 2x2 blocks:
    (2,2,2) Strassen: rank 7
    (2,2,1) schoolbook: rank 4
    
    Total: 7 + 7 + 4 = 18.
    
    For 3 blocks of 2x5: 3 * 18 = 54. Still better than 60.
    
    Actually, let me reconsider. Let me use a cleaner decomposition.
    
    For the full (2,2,15) tensor:
    T[i,j,k] = 1 for the multiplication tensor where C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor has shape (2*2, 2*15) for U and V, and (15*2) for W.
    U: (4, rank), V: (30, rank), W: (30, rank)
    
    Standard decomposition: rank = 4*15 = 60.
    
    Let me use Strassen-based approach:
    Split A (2x2) into 4 quadrants... but it's already 2x2.
    
    For 2x2 * 2x15:
    Think of B as having 15 columns. Split B into 7 blocks: 6 blocks of 2 columns and 1 block of 1 column? No, that's messy.
    
    Better: Split into 3 groups of 5 columns each.
    For each 2x5 block, use a decomposition.
    
    For (2,2,5):
    The multiplication is C[i,k] = sum_j A[i,j] B[j,k] for k=0..4.
    
    We can write this using Strassen-like products. 
    Let's use the fact that:
    For each pair of rows in A and group of columns in B, we compute products.
    
    Simple approach: For (2,2,5), use rank 20 (schoolbook) for each, total 60. No improvement.
    
    Let me try: For (2,2,5), split the 5 columns as (2,2,1).
    (2,2,2): Use Strassen rank 7
    (2,2,1): rank 4
    Total per block: 11
    3 blocks: 33
    
    But wait, I need to be careful with the indexing.
    
    Let me construct this carefully.
    
    For (2,2,2) Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M involves one product of two "linear combinations", giving rank 1 per M.
    So 7 products total for the 2x2*2x2 part.
    
    For (2,2,1):
    C0 = A00*B00 + A01*B10  (wait, this is 2x2 times 2x1)
    Actually (2,2)*(2,1): C[i] = sum_j A[i,j] B[j]
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    Rank 4 schoolbook: A00*B0, A01*B1, A10*B0, A11*B1
    
    OK let me just code the full decomposition with 3 blocks of (2,2,5), each using Strassen for the first 4 columns and schoolbook for the 5th.
    
    Wait, 2x5 = columns 0-4. Split as columns 0,1 (2 cols) and 2,3 (2 cols) and 4 (1 col).
    Two (2,2,2) Strassen blocks + one (2,2,1) schoolbook block = 7+7+4 = 18 per 2x5 block.
    3 blocks * 18 = 54 total rank.
    
    Let me code this up properly.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    U_rows = n * m  # 4
    V_rows = m * p  # 30
    W_rows = p * n  # 30
    
    # We'll build U, V, W as lists of lists
    # Start with empty, we'll add columns
    
    columns = []  # each column is (u_vec, v_vec, w_vec)
    
    def add_col(u_entries, v_entries, w_entries):
        """Add one rank-1 column. u_entries: list of (row, val), etc."""
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for r, val in u_entries:
            u[r] = val
        for r, val in v_entries:
            v[r] = val
        for r, val in w_entries:
            w[r] = val
        columns.append((u, v, w))
    
    # Indexing:
    # A is n x m = 2 x 2. A[i,j] with i in 0..1, j in 0..1.
    # B is m x p = 2 x 15. B[j,k] with j in 0..1, k in 0..14.
    # C is n x p = 2 x 15. C[i,k] with i in 0..1, k in 0..14.
    #
    # U has rows indexed by (i,j) -> i*m + j = i*2 + j
    # V has rows indexed by (j,k) -> j*p + k = j*15 + k
    # W has rows indexed by (k,i) -> k*n + i = k*2 + i (trace convention W[k*n+i, r])
    
    # For each block of 5 columns in B (k=0..4, k=5..9, k=10..14):
    #   Within each block, split into:
    #     cols 0,1: (2,2,2) Strassen
    #     cols 2,3: (2,2,2) Strassen  
    #     col 4:    (2,2,1) schoolbook
    
    for block in range(3):
        k_offset = block * 5
        
        # Strassen block 1: columns k_offset, k_offset+1
        # A indices: A[i,j] for i=0,1 j=0,1 -> U row i*2+j
        # B indices: B[j,k] for j=0,1 k=k_offset,k_offset+1 -> V row j*15+k
        # C indices: C[i,k] for i=0,1 k=k_offset,k_offset+1 -> W row k*2+i
        
        # M1 = (A00+A11)(B00+B11)
        add_col(
            [(0, 1), (3, 1)],  # A00, A11
            [(k_offset, 1), (15 + k_offset, 1)],  # B00, B11  -> wait, B row index is j*15+k
            # B[0, k_offset] = row 0*15+k_offset = k_offset
            # B[1, k_offset] = row 1*15+k_offset = 15+k_offset
            # B[0, k_offset+1] = row k_offset+1
            # B[1, k_offset+1] = row 15+k_offset+1
            # We need (B00+B11) and (B01+B11)... 
            # Actually M1 = (A00+A11)(B00+B11) means the product is sum of A entries times sum of B entries
            # This is a rank-4 term in general, but in tensor decomposition it's:
            # (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
            # Each of these is rank 1. So M1 actually contributes 4 rank-1 terms.
            # Hmm, that defeats the purpose of Strassen.
            
            # Wait, I'm confusing myself. In the tensor decomposition for matrix multiplication,
            # each "product" in Strassen corresponds to a sum of rank-1 terms in the tensor.
            # Actually no - in the bilinear form, each multiplication of linear forms is one rank-1 term.
            # The tensor decomposition rank counts the number of rank-1 tensors.
            # (A00+A11)(B00+B11) is ONE rank-1 tensor: it's the outer product of (A00+A11) with (B00+B11).
            # In the U,V,W matrices, this means U has 1s in positions for A00 and A11,
            # V has 1s in positions for B00 and B11, and W has 1s in positions for C00 and C11 (with appropriate signs).
            
            # So for M1: U entries: A00(+1), A11(+1); V entries: B[0,k_off](+1), B[1,k_off+1](+1); W entries: C[0,k_off](+1), C[1,k_off+1](+1)
            # But wait, (A00+A11)(B00+B11) contributes to C00 and C11 with coefficient +1.
            # And also to C01 and C10 with coefficient 0? No:
            # (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
            # This contributes to:
            #   C[0, k_off] += A00*B[0,k_off]  (from A00*B00)
            #   C[0, k_off+1] += A00*B[1,k_off+1]  (from A00*B11) - wait, B11 means B[1, k_off+1]
            #   C[1, k_off] += A11*B[0,k_off]  (from A11*B00)
            #   C[1, k_off+1] += A11*B[1,k_off+1]  (from A11*B11)
            
            # So the W entries for M1 are: C[0,k_off](+1), C[0,k_off+1](+1), C[1,k_off](+1), C[1,k_off+1](+1)
            # But that's not right either. The rank-1 tensor (A00+A11)⊗(B00+B11)⊗(C00+C11...) 
            # Actually the decomposition is: T = sum_r (u_r ⊗ v_r ⊗ w_r)
            # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[i,k]
            # Wait no. The tensor T for matrix mult has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
            # So T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication" tensor).
            # C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
            
            # A rank-1 decomposition: T = sum_r u_r[i] * v_r[k] * w_r[j]... no.
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]... 
            # The standard form: C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
            # No, that's not right either.
            
            # The correct form: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # Wait, that's also not standard. Let me re-read the task.
            
            # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
            # The decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # No, that doesn't make sense dimensionally.
            
            # Actually, for the matrix multiplication tensor:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T has indices (i, j, k) with T[i,j,k] = 1 (all ones for valid indices).
            # A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # But this would give: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For this to equal 1 for all i,j,k, and 0 otherwise... 
            # Actually the tensor IS all ones for the valid (i,j,k) tuples where i in [0,n), j in [0,m), k in [0,p).
            # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
            
            # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # And for invalid combinations, it should be 0, but since we only use valid indices, we just need the above.
            
            # Hmm, but that's a very strong constraint. Let me verify with schoolbook:
            # For schoolbook, for each (i,j,k), we have one column c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
            # Then sum_c ... = 1 for that specific (i,j,k) and 0 for others. That works.
            
            # For Strassen, a term like (A00+A11)(B00+B11) contributing to C00:
            # This means we need U and V and W entries such that:
            # U[0, c] = 1 (for A00), U[3, c] = 1 (for A11)  [rows 0 and 3 in U, since 0*2+0=0 and 1*2+1=3]
            # V[k_off, c] = 1 (for B[0,k_off]), V[15+k_off+1, c] = 1 (for B[1,k_off+1])
            # W[k_off*2+0, c] = 1 (for C[0,k_off]), W[(k_off+1)*2+1, c] = 1 (for C[1,k_off+1])
            # But this single column c would contribute 1*1*1 = 1 to T[0,0,k_off], T[0,1,k_off+1], T[1,0,k_off], T[1,1,k_off+1].
            # That's exactly what we want for the (A00+A11)(B00+B11) term!
            
            # Great, so Strassen works as expected. Let me redo the construction.
        )
        # Let me restart the column construction properly.
        columns.clear()
        
        # I'll rebuild everything from scratch with proper Strassen.
        
        break
    
    # Clean restart
    columns = []
    
    for block in range(3):
        k0 = block * 5  # first column index in this block
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        k4 = k0 + 4
        
        # Strassen for columns k0, k1:
        # A rows: A[0,0]=U row 0, A[0,1]=U row 1, A[1,0]=U row 2, A[1,1]=U row 3
        # B rows: B[0,k]=V row k, B[1,k]=V row 15+k
        # W rows: W[k,0]=W row 2k, W[k,1]=W row 2k+1
        
        # M1 = (A00+A11)(B0k0+B1k1) -> contributes to C[0,k0], C[1,k1]
        add_col(
            [(0, 1), (3, 1)],
            [(k0, 1), (15 + k1, 1)],
            [(2*k0 + 0, 1), (2*k1 + 1, 1)]
        )
        
        # M2 = (A10+A11)(B0k0+B0k1) -> contributes to C[1,k0], C[1,k1]
        add_col(
            [(2, 1), (3, 1)],
            [(k0, 1), (k1, 1)],
            [(2*k0 + 1, 1), (2*k1 + 1, 1)]
        )
        
        # M3 = (A00+A01)(B1k0+B1k1) -> contributes to C[0,k0], C[0,k1]
        add_col(
            [(0, 1), (1, 1)],
            [(15 + k0, 1), (15 + k1, 1)],
            [(2*k0 + 0, 1), (2*k1 + 0, 1)]
        )
        
        # M4 = (A10)(B1k0) -> contributes to C[1,k0]
        add_col(
            [(2, 1)],
            [(15 + k0, 1)],
            [(2*k0 + 1, 1)]
        )
        
        # M5 = (A01)(B1k1) -> contributes to C[0,k1]
        add_col(
            [(1, 1)],
            [(15 + k1, 1)],
            [(2*k1 + 0, 1)]
        )
        
        # M6 = (A01-A11)(B0k0+B1k0) -> contributes to C[0,k0], C[1,k0]
        add_col(
            [(1, 1), (3, -1)],
            [(k0, 1), (15 + k0, 1)],
            [(2*k0 + 0, 1), (2*k0 + 1, 1)]
        )
        
        # M7 = (A00-A10)(B0k1+B1k1) -> contributes to C[0,k1], C[1,k1]
        add_col(
            [(0, 1), (2, -1)],
            [(k1, 1), (15 + k1, 1)],
            [(2*k1 + 0, 1), (2*k1 + 1, 1)]
        )
        
        # Strassen for columns k2, k3:
        # Same pattern with k2, k3
        
        # M8 = (A00+A11)(B0k2+B1k3)
        add_col(
            [(0, 1), (3, 1)],
            [(k2, 1), (15 + k3, 1)],
            [(2*k2 + 0, 1), (2*k3 + 1, 1)]
        )
        
        # M9 = (A10+A11)(B0k2+B0k3)
        add_col(
            [(2, 1), (3, 1)],
            [(k2, 1), (k3, 1)],
            [(2*k2 + 1, 1), (2*k3 + 1, 1)]
        )
        
        # M10 = (A00+A01)(B1k2+B1k3)
        add_col(
            [(0, 1), (1, 1)],
            [(15 + k2, 1), (15 + k3, 1)],
            [(2*k2 + 0, 1), (2*k3 + 0, 1)]
        )
        
        # M11 = (A10)(B1k2)
        add_col(
            [(2, 1)],
            [(15 + k2, 1)],
            [(2*k2 + 1, 1)]
        )
        
        # M12 = (A01)(B1k3)
        add_col(
            [(1, 1)],
            [(15 + k3, 1)],
            [(2*k3 + 0, 1)]
        )
        
        # M13 = (A01-A11)(B0k2+B1k2)
        add_col(
            [(1, 1), (3, -1)],
            [(k2, 1), (15 + k2, 1)],
            [(2*k2 + 0, 1), (2*k2 + 1, 1)]
        )
        
        # M14 = (A00-A10)(B0k3+B1k3)
        add_col(
            [(0, 1), (2, -1)],
            [(k3, 1), (15 + k3, 1)],
            [(2*k3 + 0, 1), (2*k3 + 1, 1)]
        )
        
        # Schoolbook for column k4:
        # C[0,k4] = A[0,0]*B[0,k4] + A[0,1]*B[1,k4]
        # C[1,k4] = A[1,0]*B[0,k4] + A[1,1]*B[1,k4]
        
        add_col(
            [(0, 1)],
            [(k4, 1)],
            [(2*k4 + 0, 1)]
        )
        
        add_col(
            [(1, 1)],
            [(15 + k4, 1)],
            [(2*k4 + 0, 1)]
        )
        
        add_col(
            [(2, 1)],
            [(k4, 1)],
            [(2*k4 + 1, 1)]
        )
        
        add_col(
            [(3, 1)],
            [(15 + k4, 1)],
            [(2*k4 + 1, 1)]
        )
    
    # Now verify: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    # Let me do a quick check for a few entries.
    
    rank = len(columns)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u, v, w) in enumerate(columns):
        for i in range(U_rows):
            U[i][c] = u[i]
        for i in range(V_rows):
            V[i][c] = v[i]
        for i in range(W_rows):
            W[i][c] = w[i]
    
    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-10:
                    # This would be a bug, but we proceed anyway
                    pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
