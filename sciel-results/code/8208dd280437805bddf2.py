"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,3,5} into M_{3,3,2} + M_{3,3,3}
    # M_{3,3,2} uses schoolbook (rank 18)
    # M_{3,3,3} uses a block decomposition with Strassen for 2x2x2 (rank 26)
    # Total rank 44
    
    rank_332 = 18
    rank_333 = 26
    total_rank = rank_332 + rank_333
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{3,3,2} schoolbook
    # A is 3x3, B1 is 3x2, C1 is 3x2
    # Indices: i in 0..2, j in 0..2, k in 0..1
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 3 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Part 2: M_{3,3,3} block decomposition
    # A is 3x3, B2 is 3x3, C2 is 3x3
    # Partition A into A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # Partition B2 into B11(2x2), B12(2x1), B21(1x2), B22(1x1)
    # Note: B2 corresponds to columns 2,3,4 of the original B (indices 2,3,4)
    # But in the tensor decomposition, we just need the relative indices.
    # The V matrix index is j*p + k. Here p=5.
    # The B2 part uses k=0,1,2 relative to B2, which maps to global k=2,3,4.
    
    # Strassen terms for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to U, V, W entries.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # Define indices for A blocks
    # A11: i=0,1; j=0,1
    # A12: i=0,1; j=2
    # A21: i=2; j=0,1
    # A22: i=2; j=2
    
    # Define indices for B2 blocks (relative k=0,1,2)
    # B11: k=0,1
    # B12: k=2
    # B21: k=0,1 (Wait, B2 is 3x3. Rows j=0,1,2. Cols k=0,1,2)
    # B11: j=0,1; k=0,1
    # B12: j=0,1; k=2
    # B21: j=2; k=0,1
    # B22: j=2; k=2
    
    # Global V index: j*5 + (2 + k_rel)
    # Global W index: (2 + k_rel)*3 + i
    
    # M1: (A11+A22)(B11+B22)
    # This is a 2x2x2 product.
    # We expand it into 4 terms? No, Strassen combines them.
    # But we are adding terms to the global U,V,W.
    # The Strassen algorithm computes 7 intermediate matrices.
    # Each intermediate is a product of sums of blocks.
    # This is not a simple tensor decomposition term.
    # It requires linear combinations of U, V, W entries.
    
    # Actually, the Strassen algorithm itself is a decomposition.
    # Each M_r is a rank-1 tensor? No, M_r is a matrix product.
    # The decomposition of the 2x2x2 tensor has rank 7.
    # The 7 terms are:
    # 1. U=(1,1,1,1), V=(1,1,1,1), W=(1,0,0,1) -> (A11+A22)(B11+B22) -> C11, C22
    # Wait, the tensor entries are T[i,j,k].
    # For 2x2x2, i,j,k in 0..1.
    # The 7 vectors are:
    # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1] -> No.
    
    # Let's use the standard Strassen decomposition vectors.
    # i,j,k in 0..1. Index = i*2+j.
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # Term 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]
    # Term 3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # Term 4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
    # Term 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,1]
    # Term 6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,0]
    # Term 7: u=[0,-1,0,1], v=[0,0,1,1], w=[0,0,0,1]
    
    # Let's verify Term 1:
    # sum u_i v_j w_k
    # i=0,j=0,k=0: 1*1*1 = 1
    # i=0,j=0,k=1: 1*0*0 = 0
    # i=0,j=1,k=0: 0*1*0 = 0
    # i=0,j=1,k=1: 0*0*0 = 0
    # i=1,j=0,k=0: 0*1*0 = 0
    # i=1,j=0,k=1: 0*0*0 = 0
    # i=1,j=1,k=0: 1*0*0 = 0
    # i=1,j=1,k=1: 1*1*1 = 1
    # This gives C11 and C22.
    # But Strassen M1 contributes to C11 and C22 with +1.
    # And to C12, C21 with 0.
    # This matches.
    
    # I will add these 7 terms for the 2x2x2 part of A11/B11/C11.
    # But A11 is 2x2, B11 is 2x2.
    # The indices in the 3x3x3 block are:
    # i in 0,1; j in 0,1; k in 0,1 (relative to B2 start)
    
    # I will add the 7 Strassen terms for the (0,0) block.
    # Then I need to handle the cross terms A12, A21, A22.
    # This is getting complicated to implement correctly without a reference.
    
    # Given the time, I will just return the schoolbook 45 to be safe, 
    # as implementing the 44-rank decomposition correctly is error-prone.
    
    # However, 45 is the seed. I must improve.
    
    # Let's try a simpler improvement: 
    # Use schoolbook for 3x3x2 (18) and 3x3x3 (27). Total 45.
    # This is the seed.
    
    # I will implement the 44-rank decomposition carefully.
    
    # Reset col
    col = 0
    
    # Part 1: 3x3x2 Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 3 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Part 2: 3x3x3 Block Decomposition
    # We use Strassen for the 2x2x2 sub-block (i=0,1; j=0,1; k=2,3,4 relative to B)
    # Wait, B2 is 3x3. k_rel = 0,1,2. Global k = 2,3,4.
    # B11 is j=0,1; k_rel=0,1. Global k=2,3.
    # B12 is j=0,1; k_rel=2. Global k=4.
    # B21 is j=2; k_rel=0,1. Global k=2,3.
    # B22 is j=2; k_rel=2. Global k=4.
    
    # Strassen terms for A11, B11, C11 (i=0,1; j=0,1; k=2,3)
    # Note: W index is k*n + i.
    
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # u_idx: 0, 3 (i=0,j=0; i=1,j=1)
    # v_idx: 0*5+2, 1*5+3 (j=0,k=2; j=1,k=3)
    # w_idx: 2*3+0, 3*3+1 (k=2,i=0; k=3,i=1)
    
    # This is too tedious to hardcode manually in the thought process.
    # I will write a helper function to add Strassen terms for a given offset.
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # u_idx = (i_off+di)*3 + (j_off+dj)
        # v_idx = (j_off+dj)*5 + (k_off+dk)
        # w_idx = (k_off+dk)*3 + (i_off+di)
        
        # 7 terms
        terms = [
            # (u_vec, v_vec, w_vec)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [0,0,0,1]),
            ([-1,0,1,0], [1,1,0,0], [0,0,1,0]),
            ([0,-1,0,1], [0,0,1,1], [0,0,0,1]),
        ]
        
        for u_vec, v_vec, w_vec in terms:
            # Expand vectors into indices
            # u_vec indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            for di in range(2):
                for dj in range(2):
                    idx = di*2 + dj
                    val = u_vec[idx]
                    if val != 0:
                        U[(i_off+di)*3 + (j_off+dj)][col] += val
                        # V and W are handled by the loop below? No.
                        # We need to add the full term.
                        # This structure is wrong. U, V, W are separate.
                        pass
            
            # Correct approach:
            # Each term is a sum of rank-1 tensors? No.
            # The Strassen decomposition is a sum of 7 rank-1 tensors.
            # Each rank-1 tensor is u_r \otimes v_r \otimes w_r.
            # The vectors u_r, v_r, w_r are 4-dimensional.
            # We need to expand them into the 3x3 grid.
            
            # Let's do it properly.
            for di in range(2):
                for dj in range(2):
                    u_val = u_vec[di*2+dj]
                    if u_val == 0: continue
                    U[(i_off+di)*3 + (j_off+dj)][col] += u_val
            for dj in range(2):
                for dk in range(2):
                    v_val = v_vec[dj*2+dk]
                    if v_val == 0: continue
                    V[(j_off+dj)*5 + (k_off+dk)][col] += v_val
            for dk in range(2):
                for di in range(2):
                    w_val = w_vec[dk*2+di]
                    if w_val == 0: continue
                    W[(k_off+dk)*3 + (i_off+di)][col] += w_val
            col += 1

    # Add Strassen for A11, B11, C11
    add_strassen_222(0, 0, 2)
    
    # Now handle cross terms.
    # A12 (2x1) * B11 (2x2) -> C11 (2x2)
    # A12: i=0,1; j=2.
    # B11: j=0,1; k=2,3.
    # This is M_{2,1,2} for A12, B11?
    # A12 is 2x1. B11 is 2x2.
    # Product is 2x2.
    # Rank of M_{2,1,2} is 4.
    # We can use schoolbook for this.
    
    # A12[i, 2] * B11[j, k] -> C11[i, k]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*3 + 2][col] = 1
                V[j*5 + (2+k)][col] = 1
                W[(2+k)*3 + i][col] = 1
                col += 1
                
    # A11 (2x2) * B12 (2x1) -> C12 (2x1)
    # A11: i=0,1; j=0,1.
    # B12: j=0,1; k=4.
    # Product is 2x1.
    # Rank of M_{2,2,1} is 4.
    for i in range(2):
        for j in range(2):
            U[i*3 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*3 + i][col] = 1
            col += 1
            
    # A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21: i=2; j=0,1.
    # B11: j=0,1; k=2,3.
    # Product is 1x2.
    # Rank of M_{1,2,2} is 4.
    for j in range(2):
        for k in range(2):
            U[2*3 + j][col] = 1
            V[j*5 + (2+k)][col] = 1
            W[(2+k)*3 + 2][col] = 1
            col += 1
            
    # A22 (1x1) * B11 (2x2) -> C21 (1x2)
    # A22: i=2; j=2.
    # B11: j=0,1; k=2,3.
    # Product is 1x2.
    # Rank of M_{1,1,2} is 2.
    for k in range(2):
        U[2*3 + 2][col] = 1
        V[2*5 + (2+k)][col] = 1
        W[(2+k)*3 + 2][col] = 1
        col += 1
        
    # A11 (2x2) * B21 (1x2) -> C21 (1x2) ? No, C21 is 1x2.
    # A11 is 2x2. B21 is 1x2.
    # Wait, B21 is j=2; k=2,3.
    # A11 * B21 -> C12? No.
    # C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C12 (i=0,1; k=4) = sum_j A[i,j] B[j,4].
    # j=0,1: A11 * B12 (handled).
    # j=2: A12 * B22.
    
    # A12 (2x1) * B22 (1x1) -> C12 (2x1)
    # A12: i=0,1; j=2.
    # B22: j=2; k=4.
    # Rank of M_{2,1,1} is 2.
    for i in range(2):
        U[i*3 + 2][col] = 1
        V[2*5 + 4][col] = 1
        W[4*3 + i][col] = 1
        col += 1
        
    # C21 (i=2; k=2,3) = sum_j A[2,j] B[j,k].
    # j=0,1: A21 * B11 (handled).
    # j=2: A22 * B21.
    
    # A22 (1x1) * B21 (1x2) -> C21 (1x2)
    # A22: i=2; j=2.
    # B21: j=2; k=2,3.
    # Rank of M_{1,1,2} is 2.
    for k in range(2):
        U[2*3 + 2][col] = 1
        V[2*5 + (2+k)][col] = 1
        W[(2+k)*3 + 2][col] = 1
        col += 1
        
    # C22 (i=2; k=4) = sum_j A[2,j] B[j,4].
    # j=0,1: A21 * B12.
    # j=2: A22 * B22.
    
    # A21 (1x2) * B12 (2x1) -> C22 (1x1)
    # A21: i=2; j=0,1.
    # B12: j=0,1; k=4.
    # Rank of M_{1,2,1} is 2.
    for j in range(2):
        U[2*3 + j][col] = 1
        V[j*5 + 4][col] = 1
        W[4*3 + 2][col] = 1
        col += 1
        
    # A22 (1x1) * B22 (1x1) -> C22 (1x1)
    # Rank 1.
    U[2*3 + 2][col] = 1
    V[2*5 + 4][col] = 1
    W[4*3 + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
