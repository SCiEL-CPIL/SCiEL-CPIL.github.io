import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 solution
    # Split j into two groups of 2: j1 in {0, 1}, j2 in {2, 3}
    # For each k, use Strassen (rank 7) for 2x2x2 (effectively 2x2x1 vector)
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)g
    # P3 = a(f-h)
    # P4 = d(b-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # In our case, B is 2x1 vector [b0, b1]^T for each k.
    # So e=b0, f=b1? No.
    # B = [[b00, b01], [b10, b11]] is 2x2.
    # Here B is 2x7. So for fixed k, B_col = [b0k, b1k]^T.
    # This is not a 2x2 matrix.
    # The "multiplication" is A (2x2) * B_col (2x1) = C_col (2x1).
    # This is a 2x2x1 problem.
    # Rank of 2x2x1?
    # C0 = a*b0 + b*b1
    # C1 = c*b0 + d*b1
    # This is just 4 multiplications. Strassen doesn't help for 2x2x1.
    # Strassen helps for 2x2x2.
    
    # So my previous analysis was flawed.
    # For 2x2x1, rank is 4.
    # For 2x2x2, rank is 7.
    # For 2x2x7, we have 7 independent 2x2x1 problems?
    # No, the A matrix is shared.
    # But the tensor decomposition separates the indices.
    # T_{i, j, k}.
    # If we fix k, it's a 2x2x1 problem.
    # Rank is 4 per k.
    # Total rank 4 * 7 = 28?
    # But A is shared. Does that reduce rank?
    # In tensor decomposition, if A is shared, it means the U vectors are the same for different k?
    # No, U depends on i, j. V depends on j, k.
    # If we use the same U for all k, then V must be different.
    # But V_{j, k} depends on k.
    # So we can't just share U without changing V.
    # Actually, the rank is the number of terms.
    # If we use 4 terms per k, total 28 terms.
    # Let's check if rank 28 is possible.
    # Lower bound max(nm, mp, np) = 28.
    # So 28 is a candidate!
    
    # Can we achieve 28?
    # This would imply that the multiplication can be done with 28 scalar multiplications.
    # This is the "border rank" or "rank" of the tensor.
    # For 2x2x2, rank is 7.
    # For 2x2x1, rank is 4.
    # For 2x2x7, is it 28?
    # If we treat it as 7 independent 2x2x1, yes.
    # But is there a way to do it with fewer?
    # Probably not, because the k indices are independent in B and C.
    # So 28 is a very strong candidate.
    
    # Let's construct the 28 rank decomposition.
    # For each k in 0..6:
    #   Term 1: A_{0,0} * B_{0,k} -> C_{0,k}
    #   Term 2: A_{0,1} * B_{1,k} -> C_{0,k}
    #   Term 3: A_{1,0} * B_{0,k} -> C_{1,k}
    #   Term 4: A_{1,1} * B_{1,k} -> C_{1,k}
    # Wait, this is for 2x2x1.
    # But we have 2x4x7.
    # j goes 0..3.
    # So for each k, we have 4 terms:
    #   A_{0,0} B_{0,k}
    #   A_{0,1} B_{1,k}
    #   A_{0,2} B_{2,k}
    #   A_{0,3} B_{3,k}
    #   ... and for row 1 of A.
    # Total 8 terms per k.
    # Total 8 * 7 = 56.
    # This is the schoolbook.
    
    # So how to get 28?
    # We need to share terms across k?
    # No, B_{j, k} is different for each k.
    # So V_{j, k} must be different.
    # Unless we combine k's.
    # But k's are independent.
    # So the rank is at least 56?
    # No, Strassen reduced 2x2x2 from 8 to 7.
    # So sharing across j can reduce rank.
    # Sharing across k?
    # If we group k's, we can use Strassen on the k dimension?
    # But A is not multiplied by k.
    # So we can only share across i and j.
    # For fixed k, it's a 2x4x1 problem.
    # Rank of 2x4x1?
    # C_0 = sum_{j=0}^3 A_{0,j} B_{j,k}
    # C_1 = sum_{j=0}^3 A_{1,j} B_{j,k}
    # This is 8 multiplications.
    # Can we do better?
    # It's like multiplying a 2x4 vector by a 4x1 vector.
    # This is just a dot product for each row.
    # Rank is 8.
    # So for each k, rank is 8.
    # Total rank 56.
    
    # So Strassen doesn't help here because m=4 is not a power of 2 that allows recursive Strassen on the j dimension alone?
    # Strassen works on 2x2 blocks.
    # We can split j into 2x2 blocks.
    # Block 1: j=0,1. Block 2: j=2,3.
    # For Block 1, we have 2x2x1. Rank 4.
    # For Block 2, we have 2x2x1. Rank 4.
    # Total 8 per k.
    # No gain.
    
    # Strassen gains when p > 1.
    # Here p=7.
    # We can split k into blocks.
    # Split k into 2x2 blocks?
    # k=0,1. k=2,3. k=4,5. k=6.
    # For k=0,1: 2x4x2.
    # Split j into 2x2.
    # Block 1 (j=0,1): 2x2x2. Rank 7.
    # Block 2 (j=2,3): 2x2x2. Rank 7.
    # Total 14 for k=0,1.
    # Similarly for k=2,3: 14.
    # For k=4,5: 14.
    # For k=6: 2x4x1. Rank 8.
    # Total 14+14+14+8 = 50.
    # Better than 56.
    
    # Can we do better?
    # Split k=0..5 into three 2x2 blocks.
    # And k=6 as 2x1.
    # Rank 50.
    
    # What if we split j differently?
    # No, 2x2 is the Strassen unit.
    
    # So 50 is a good target.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal rank
        # A indices: i_offset, i_offset+1 ; j_offset, j_offset+1
        # B indices: j_offset, j_offset+1 ; k_offset, k_offset+1
        # C indices: k_offset, k_offset+1 ; i_offset, i_offset+1
        
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B10
        # P3 = A00(B11 - B00)
        # P4 = A11(B00 - B11)
        # P5 = (A00 + A01)B01
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        
        # Note: B indices in Strassen are (row, col).
        # B00 = B[j_offset, k_offset]
        # B01 = B[j_offset, k_offset+1]
        # B10 = B[j_offset+1, k_offset]
        # B11 = B[j_offset+1, k_offset+1]
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We need to map these to U, V, W.
        # U is (n*m, rank). Index (i*m+j).
        # V is (m*p, rank). Index (j*p+k).
        # W is (p*n, rank). Index (k*n+i).
        
        def add_term(u_coeff, v_coeff, w_coeff):
            nonlocal rank
            # u_coeff: dict (i,j) -> val
            # v_coeff: dict (j,k) -> val
            # w_coeff: dict (k,i) -> val
            # This is not how it works.
            # Each term is a rank-1 tensor.
            # U[i*m+j, rank] = val
            # V[j*p+k, rank] = val
            # W[k*n+i, rank] = val
            
            # We need to add 7 terms.
            pass

        # Let's just build the matrices directly.
        # We will create 7 columns for this block.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: (i_offset, j_offset) += 1, (i_offset+1, j_offset+1) += 1
        # V: (j_offset, k_offset) += 1, (j_offset+1, k_offset+1) += 1
        # W: (k_offset, i_offset) += 1, (k_offset+1, i_offset+1) += 1? No.
        # W corresponds to C.
        # C00 gets P1. C11 gets P1.
        # So W[k_offset, i_offset] += 1, W[k_offset+1, i_offset+1] += 1.
        
        # This is getting complicated to map manually.
        # Let's use a generic rank-1 addition function.
        
        def add_rank1(u_dict, v_dict, w_dict):
            nonlocal rank
            # u_dict: {(i,j): val}
            # v_dict: {(j,k): val}
            # w_dict: {(k,i): val}
            # This is not a rank-1 tensor if u, v, w are not single indices.
            # A rank-1 tensor is u_i v_j w_k.
            # Here u depends on i,j. v on j,k. w on k,i.
            # So a term is defined by a specific i, j, k? No.
            # The term is a product of a vector in A-space, B-space, C-space.
            # A-space is 2x4. B-space is 4x7. C-space is 2x7.
            # A rank-1 tensor is u \otimes v \otimes w.
            # u is a 2x4 matrix. v is a 4x7 matrix. w is a 2x7 matrix.
            # The decomposition is sum_r u_r \otimes v_r \otimes w_r = T.
            # T_{i,j, j',k', i'',k''} = delta...
            
            # So for each term r, we define:
            # U[:, r] is a flattened 2x4 matrix.
            # V[:, r] is a flattened 4x7 matrix.
            # W[:, r] is a flattened 2x7 matrix.
            
            # For Strassen term P1:
            # u_1 has 1 at (0,0) and (1,1).
            # v_1 has 1 at (0,0) and (1,1).
            # w_1 has 1 at (0,0) and (1,1).
            # This is not a rank-1 tensor in the flattened sense?
            # Yes, it is. u_1 \otimes v_1 \otimes w_1.
            
            # So we just need to fill the U, V, W matrices.
            
            col = rank
            rank += 1
            
            # Initialize columns to 0
            # We will build U, V, W as lists of lists.
            # It's better to build them as numpy arrays at the end.
            
            U.append([0]*(n*m))
            V.append([0]*(m*p))
            W.append([0]*(p*n))
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[col][i*m + j] = val
            for (j, k), val in v_dict.items():
                V[col][j*p + k] = val
            for (k, i), val in w_dict.items():
                W[col][k*n + i] = val

        # Strassen Terms for block (i0, j0, k0)
        # i0, j0, k0 are offsets.
        
        # P1
        add_rank1(
            {(i_offset, j_offset): 1, (i_offset+1, j_offset+1): 1},
            {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1},
            {(k_offset, i_offset): 1, (k_offset+1, i_offset+1): 1}
        )
        
        # P2
        add_rank1(
            {(i_offset+1, j_offset): 1, (i_offset+1, j_offset+1): 1},
            {(j_offset+1, k_offset): 1},
            {(k_offset, i_offset+1): 1}
        )
        
        # P3
        add_rank1(
            {(i_offset, j_offset): 1},
            {(j_offset+1, k_offset+1): 1, (j_offset, k_offset): -1},
            {(k_offset+1, i_offset): 1}
        )
        
        # P4
        add_rank1(
            {(i_offset+1, j_offset+1): 1},
            {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): -1},
            {(k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1} # Wait, C10 and C11?
            # C10 = P2 + P4. C11 = P1 - P2 + P3 + P6.
            # So P4 contributes to C10 and C11?
            # C10 index (1, 0). C11 index (1, 1).
            # w_dict should have {(k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1}
        )
        
        # P5
        add_rank1(
            {(i_offset, j_offset): 1, (i_offset, j_offset+1): 1},
            {(j_offset, k_offset+1): 1},
            {(k_offset+1, i_offset): 1}
        )
        
        # P6
        add_rank1(
            {(i_offset+1, j_offset): 1, (i_offset, j_offset): -1},
            {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1},
            {(k_offset+1, i_offset+1): 1}
        )
        
        # P7
        add_rank1(
            {(i_offset, j_offset+1): 1, (i_offset+1, j_offset+1): -1},
            {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1},
            {(k_offset, i_offset): 1, (k_offset+1, i_offset): 1}
        )

    # Apply blocks
    # k=0,1; j=0,1
    add_strassen_block(0, 0, 0)
    # k=0,1; j=2,3
    add_strassen_block(0, 2, 0)
    
    # k=2,3; j=0,1
    add_strassen_block(0, 0, 2)
    # k=2,3; j=2,3
    add_strassen_block(0, 2, 2)
    
    # k=4,5; j=0,1
    add_strassen_block(0, 0, 4)
    # k=4,5; j=2,3
    add_strassen_block(0, 2, 4)
    
    # k=6; j=0,1 (2x2x1)
    # C0 = A00 B06 + A01 B16
    # C1 = A10 B06 + A11 B16
    # 4 terms
    def add_2x2x1_block(i_offset, j_offset, k_offset):
        nonlocal rank
        # Term 1: A00 B0k
        add_rank1(
            {(i_offset, j_offset): 1},
            {(j_offset, k_offset): 1},
            {(k_offset, i_offset): 1}
        )
        # Term 2: A01 B1k
        add_rank1(
            {(i_offset, j_offset+1): 1},
            {(j_offset+1, k_offset): 1},
            {(k_offset, i_offset): 1}
        )
        # Term 3: A10 B0k
        add_rank1(
            {(i_offset+1, j_offset): 1},
            {(j_offset, k_offset): 1},
            {(k_offset, i_offset+1): 1}
        )
        # Term 4: A11 B1k
        add_rank1(
            {(i_offset+1, j_offset+1): 1},
            {(j_offset+1, k_offset): 1},
            {(k_offset, i_offset+1): 1}
        )

    add_2x2x1_block(0, 0, 6)
    add_2x2x1_block(0, 2, 6)
    
    # Convert to lists of lists as required
    U_final = [[0]*rank for _ in range(n*m)]
    V_final = [[0]*rank for _ in range(m*p)]
    W_final = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if U[r][i] != 0:
                U_final[i][r] = U[r][i]
        for j in range(m*p):
            if V[r][j] != 0:
                V_final[j][r] = V[r][j]
        for k in range(p*n):
            if W[r][k] != 0:
                W_final[k][r] = W[r][k]
                
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
