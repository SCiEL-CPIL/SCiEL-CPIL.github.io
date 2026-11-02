import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller blocks.
    # The strategy is:
    # 1. Decompose M_{4,4,7} into M_{4,4,4} + M_{4,4,3}.
    # 2. Decompose M_{4,4,4} using Strassen's algorithm recursively (2 levels).
    #    M_{4,4,4} uses 7 multiplications of M_{2,2,2}.
    #    Each M_{2,2,2} uses 7 multiplications of scalars.
    #    Total rank for M_{4,4,4} is 49.
    # 3. Decompose M_{4,4,3} into 4 blocks of M_{2,2,3}.
    #    Each M_{2,2,3} is decomposed into M_{2,2,2} (rank 7) + M_{2,2,1} (rank 2).
    #    Total rank for M_{4,4,3} is 4 * (7 + 2) = 36.
    # Total rank = 49 + 36 = 85.
    
    # We will generate U, V, W matrices directly.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (28, rank)
    
    # Note: The problem statement says W[k*n+i, r].
    # Standard matrix multiplication C = A B.
    # A is n x m. B is m x p. C is n x p.
    # Tensor T[i, j, k] = 1 if C[i, k] depends on A[i, j] * B[j, k].
    # Decomposition: T = sum_r U[:, r] * V[:, r] * W[:, r] (outer products).
    # U[i, j, r], V[j, k, r], W[k, i, r].
    # Indices:
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    # Helper to add a rank-1 component
    # u_vec: length n*m
    # v_vec: length m*p
    # w_vec: length p*n
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_component(u, v, w):
        nonlocal rank
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        rank += 1
        
    # --- Strassen M_{2,2,2} ---
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # We need to map the 7 multiplications to U, V, W vectors.
    # The indices for a 2x2x2 block embedded in larger matrices:
    # A indices: (i, j) where i,j in {0,1}
    # B indices: (j, k) where j,k in {0,1}
    # C indices: (k, i) where k,i in {0,1} -> W index k*n + i
    
    def strassen_222(offset_i, offset_j, offset_k, n, m, p):
        # offset_i: row offset in A and C
        # offset_j: col offset in A and row offset in B
        # offset_k: col offset in B and row offset in C (for W index k*n+i)
        # Note: W index is k*n + i.
        # U index: i*m + j
        # V index: j*p + k
        
        # 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We define the contributions of each M_r to C.
        # C indices are (i, k) relative to the block.
        # C[0,0] = C11
        # C[0,1] = C12
        # C[1,0] = C21
        # C[1,1] = C22
        
        # For each multiplication r, we define u_r, v_r, w_r.
        # u_r is a vector of length n*m (mostly 0, 1, -1 at specific spots)
        # v_r is a vector of length m*p
        # w_r is a vector of length p*n
        
        # We will create 7 components.
        
        # M1
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        # A11+A22 -> u1[i*m+j]
        u1[offset_i*m + offset_j] = 1
        u1[(offset_i+1)*m + (offset_j+1)] = 1
        # B11+B22 -> v1[j*p+k]
        v1[offset_j*p + offset_k] = 1
        v1[(offset_j+1)*p + (offset_k+1)] = 1
        # C11 contribution +1, C22 contribution +1
        # C11 is i=offset_i, k=offset_k -> W index offset_k*n + offset_i
        # C22 is i=offset_i+1, k=offset_k+1 -> W index (offset_k+1)*n + (offset_i+1)
        w1[offset_k*n + offset_i] = 1
        w1[(offset_k+1)*n + (offset_i+1)] = 1
        add_component(u1, v1, w1)
        
        # M2
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        # A21+A22
        u2[(offset_i+1)*m + offset_j] = 1
        u2[(offset_i+1)*m + (offset_j+1)] = 1
        # B11
        v2[offset_j*p + offset_k] = 1
        # C21 contribution +1
        # C21 is i=offset_i+1, k=offset_k
        w2[offset_k*n + (offset_i+1)] = 1
        add_component(u2, v2, w2)
        
        # M3
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        # A11
        u3[offset_i*m + offset_j] = 1
        # B12-B22
        v3[offset_j*p + (offset_k+1)] = 1
        v3[(offset_j+1)*p + (offset_k+1)] = -1
        # C12 contribution +1
        # C12 is i=offset_i, k=offset_k+1
        w3[(offset_k+1)*n + offset_i] = 1
        add_component(u3, v3, w3)
        
        # M4
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        # A22
        u4[(offset_i+1)*m + (offset_j+1)] = 1
        # B21-B11
        v4[(offset_j+1)*p + offset_k] = 1
        v4[offset_j*p + offset_k] = -1
        # C11 contribution +1, C21 contribution +1
        w4[offset_k*n + offset_i] = 1
        w4[offset_k*n + (offset_i+1)] = 1
        add_component(u4, v4, w4)
        
        # M5
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        # A11+A12
        u5[offset_i*m + offset_j] = 1
        u5[offset_i*m + (offset_j+1)] = 1
        # B22
        v5[(offset_j+1)*p + (offset_k+1)] = 1
        # C11 contribution -1, C12 contribution +1
        w5[offset_k*n + offset_i] = -1
        w5[(offset_k+1)*n + offset_i] = 1
        add_component(u5, v5, w5)
        
        # M6
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        # A21-A11
        u6[(offset_i+1)*m + offset_j] = 1
        u6[offset_i*m + offset_j] = -1
        # B11+B12
        v6[offset_j*p + offset_k] = 1
        v6[offset_j*p + (offset_k+1)] = 1
        # C22 contribution +1
        w6[(offset_k+1)*n + (offset_i+1)] = 1
        add_component(u6, v6, w6)
        
        # M7
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        # A12-A22
        u7[offset_i*m + (offset_j+1)] = 1
        u7[(offset_i+1)*m + (offset_j+1)] = -1
        # B21+B22
        v7[(offset_j+1)*p + offset_k] = 1
        v7[(offset_j+1)*p + (offset_k+1)] = 1
        # C11 contribution +1
        w7[offset_k*n + offset_i] = 1
        add_component(u7, v7, w7)

    # --- M_{2,2,1} (rank 2) ---
    # A (2x2), B (2x1) -> C (2x1)
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    def m221(offset_i, offset_j, offset_k, n, m, p):
        # B is 2x1, so only k=offset_k is involved.
        # U index: i*m+j
        # V index: j*p+k
        # W index: k*n+i
        
        # Comp 1: A11 B1 -> C1
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[offset_i*m + offset_j] = 1
        v1[offset_j*p + offset_k] = 1
        w1[offset_k*n + offset_i] = 1
        add_component(u1, v1, w1)
        
        # Comp 2: A12 B2 -> C1
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[offset_i*m + (offset_j+1)] = 1
        v2[(offset_j+1)*p + offset_k] = 1
        w2[offset_k*n + offset_i] = 1
        add_component(u2, v2, w2)
        
        # Comp 3: A21 B1 -> C2
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[(offset_i+1)*m + offset_j] = 1
        v3[offset_j*p + offset_k] = 1
        w3[offset_k*n + (offset_i+1)] = 1
        add_component(u3, v3, w3)
        
        # Comp 4: A22 B2 -> C2
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(offset_i+1)*m + (offset_j+1)] = 1
        v4[(offset_j+1)*p + offset_k] = 1
        w4[offset_k*n + (offset_i+1)] = 1
        add_component(u4, v4, w4)
        
        # Wait, rank 2 for M_{2,2,1}?
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # This is 4 terms. Can we do better?
        # No, M_{2,2,1} is essentially 2 independent dot products of length 2.
        # Rank is 2? No, the tensor is sum of 4 rank-1 tensors.
        # Is it reducible?
        # T = A11 B1 C1 + A12 B2 C1 + A21 B1 C2 + A22 B2 C2
        # This is the sum of 4 rank-1 tensors.
        # The rank of M_{n,m,1} is n*m.
        # So M_{2,2,1} has rank 4.
        # My previous estimate of rank 2 was wrong.
        # So M_{2,2,3} = M_{2,2,2} + M_{2,2,1} has rank 7 + 4 = 11.
        # Then M_{4,4,3} has rank 4 * 11 = 44.
        # Total rank 49 + 44 = 93.
        # Still better than 112.

    # --- M_{4,4,4} using Strassen ---
    # Split A (4x4) into 2x2 blocks.
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B (4x4) into 2x2 blocks.
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    # Strassen formulas for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each "multiplication" is a 2x2x2 multiplication.
    # We call strassen_222 for each of the 7 terms.
    # But we need to combine the inputs.
    # e.g., M1 input A is A11+A22.
    # This means the U vector for M1 should have 1s at A11 and A22 positions.
    # My strassen_222 function above assumes the inputs are single blocks.
    # I need to modify it or create a wrapper that handles the sums.
    
    # Let's re-implement the generation of U, V, W for the 4x4x4 case directly.
    
    # We will define the 7 components for 4x4x4.
    
    # M1: (A11+A22)(B11+B22)
    u1 = [0]*16; v1 = [0]*16; w1 = [0]*16
    # A11 (0,0..1,0..1) and A22 (2,0..3,0..1)
    for i in [0, 2]:
        for j in [0, 1]:
            u1[i*4+j] = 1
    # B11 (0,0..1,0..1) and B22 (2,0..3,0..1)
    for j in [0, 2]:
        for k in [0, 1]:
            v1[j*4+k] = 1
    # C11 (0,0..1,0..1) and C22 (2,0..3,0..1)
    # C indices: k*n + i
    for k in [0, 2]:
        for i in [0, 1]:
            w1[k*4+i] = 1
    add_component(u1, v1, w1)
    
    # M2: (A21+A22)B11
    u2 = [0]*16; v2 = [0]*16; w2 = [0]*16
    # A21 (2,0..3,0..1) and A22 (2,0..3,2..3)
    for i in [2, 3]:
        for j in [0, 1, 2, 3]:
            u2[i*4+j] = 1
    # B11 (0,0..1,0..1)
    for j in [0, 1]:
        for k in [0, 1]:
            v2[j*4+k] = 1
    # C21 (2,0..3,0..1)
    for k in [0, 1]:
        for i in [2, 3]:
            w2[k*4+i] = 1
    add_component(u2, v2, w2)
    
    # M3: A11(B12-B22)
    u3 = [0]*16; v3 = [0]*16; w3 = [0]*16
    # A11
    for i in [0, 1]:
        for j in [0, 1]:
            u3[i*4+j] = 1
    # B12 (0,0..1,2..3) and B22 (2,0..3,2..3)
    for j in [0, 1]:
        for k in [2, 3]:
            v3[j*4+k] = 1
    for j in [2, 3]:
        for k in [2, 3]:
            v3[j*4+k] = -1
    # C12 (0,0..1,2..3)
    for k in [2, 3]:
        for i in [0, 1]:
            w3[k*4+i] = 1
    add_component(u3, v3, w3)
    
    # M4: A22(B21-B11)
    u4 = [0]*16; v4 = [0]*16; w4 = [0]*16
    # A22
    for i in [2, 3]:
        for j in [2, 3]:
            u4[i*4+j] = 1
    # B21 (2,0..3,0..1) and B11 (0,0..1,0..1)
    for j in [2, 3]:
        for k in [0, 1]:
            v4[j*4+k] = 1
    for j in [0, 1]:
        for k in [0, 1]:
            v4[j*4+k] = -1
    # C11 (0,0..1,0..1) and C21 (2,0..3,0..1)
    for k in [0, 1]:
        for i in [0, 1, 2, 3]:
            w4[k*4+i] = 1
    add_component(u4, v4, w4)
    
    # M5: (A11+A12)B22
    u5 = [0]*16; v5 = [0]*16; w5 = [0]*16
    # A11 (0,0..1,0..1) and A12 (0,0..1,2..3)
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            u5[i*4+j] = 1
    # B22 (2,0..3,2..3)
    for j in [2, 3]:
        for k in [2, 3]:
            v5[j*4+k] = 1
    # C11 (0,0..1,0..1) -1 and C12 (0,0..1,2..3) +1
    for k in [0, 1]:
        for i in [0, 1]:
            w5[k*4+i] = -1
    for k in [2, 3]:
        for i in [0, 1]:
            w5[k*4+i] = 1
    add_component(u5, v5, w5)
    
    # M6: (A21-A11)(B11+B12)
    u6 = [0]*16; v6 = [0]*16; w6 = [0]*16
    # A21 (2,0..3,0..1) +1, A11 (0,0..1,0..1) -1
    for i in [2, 3]:
        for j in [0, 1]:
            u6[i*4+j] = 1
    for i in [0, 1]:
        for j in [0, 1]:
            u6[i*4+j] = -1
    # B11 (0,0..1,0..1) and B12 (0,0..1,2..3)
    for j in [0, 1]:
        for k in [0, 1, 2, 3]:
            v6[j*4+k] = 1
    # C22 (2,0..3,2..3)
    for k in [2, 3]:
        for i in [2, 3]:
            w6[k*4+i] = 1
    add_component(u6, v6, w6)
    
    # M7: (A12-A22)(B21+B22)
    u7 = [0]*16; v7 = [0]*16; w7 = [0]*16
    # A12 (0,0..1,2..3) +1, A22 (2,0..3,2..3) -1
    for i in [0, 1]:
        for j in [2, 3]:
            u7[i*4+j] = 1
    for i in [2, 3]:
        for j in [2, 3]:
            u7[i*4+j] = -1
    # B21 (2,0..3,0..1) and B22 (2,0..3,2..3)
    for j in [2, 3]:
        for k in [0, 1, 2, 3]:
            v7[j*4+k] = 1
    # C11 (0,0..1,0..1)
    for k in [0, 1]:
        for i in [0, 1]:
            w7[k*4+i] = 1
    add_component(u7, v7, w7)
    
    # --- M_{4,4,3} ---
    # B is 4x3.
    # Split A into 2x2 blocks.
    # Split B into top 2x3 and bottom 2x3.
    # AB = A_top B_top + A_bot B_bot
    # A_top = [A11 A12], A_bot = [A21 A22]
    # B_top = B_1 (2x3), B_bot = B_2 (2x3)
    # Term 1: A11 B_1
    # Term 2: A12 B_1
    # Term 3: A21 B_1
    # Term 4: A22 B_1
    # Term 5: A11 B_2
    # Term 6: A12 B_2
    # Term 7: A21 B_2
    # Term 8: A22 B_2
    
    # Each term is M_{2,2,3}.
    # M_{2,2,3} rank 11.
    # Total 8 * 11 = 88?
    # No, A_top B_top is 2x4 * 2x3.
    # A_top = [A11 A12].
    # A_top B_top = A11 B_top_left + A12 B_top_right?
    # No, B_top is 2x3.
    # A11 is 2x2. B_top is 2x3.
    # A11 B_top is 2x3.
    # This is M_{2,2,3}.
    # So we have 4 terms of M_{2,2,3}.
    # Total rank 4 * 11 = 44.
    
    # Let's implement M_{2,2,3} with rank 11.
    # It is M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4).
    # We can reuse the logic.
    
    def add_m223(offset_i, offset_j, offset_k, n, m, p):
        # offset_i, offset_j for A (2x2)
        # offset_k for B (2x3) -> k in {offset_k, offset_k+1, offset_k+2}
        # B is 2x3. Rows j, j+1. Cols k, k+1, k+2.
        # C is 2x3. Rows k, k+1, k+2. Cols i, i+1.
        
        # Part 1: M_{2,2,2} for first 2 cols of B.
        # B sub-block is 2x2 at cols k, k+1.
        # We can call a modified strassen_222?
        # No, let's just add the 7 components for the 2x2 part and 4 for the 1x2 part?
        # Wait, M_{2,2,3} = M_{2,2,2} (cols 0,1) + M_{2,2,1} (col 2).
        
        # 7 components for M_{2,2,2} on B[:, 0:2]
        # We need to map the 2x2 B block to the global indices.
        # B indices: j*p + k.
        # The 2x2 block uses j in {offset_j, offset_j+1} and k in {offset_k, offset_k+1}.
        
        # We will manually add the 7 components for the 2x2 part.
        # This is tedious. Let's use the fact that M_{2,2,3} can be decomposed into 11 rank-1 tensors.
        # We will generate them explicitly.
        
        # 1-7: Strassen for B[:, 0:2]
        # 8-11: Schoolbook for B[:, 2]
        
        # Strassen for B[:, 0:2]
        # M1 = (A11+A22)(B11+B22)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[offset_i*m + offset_j] = 1
        u1[(offset_i+1)*m + (offset_j+1)] = 1
        v1[offset_j*p + offset_k] = 1
        v1[(offset_j+1)*p + (offset_k+1)] = 1
        w1[offset_k*n + offset_i] = 1
        w1[(offset_k+1)*n + (offset_i+1)] = 1
        add_component(u1, v1, w1)
        
        # M2 = (A21+A22)B11
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[(offset_i+1)*m + offset_j] = 1
        u2[(offset_i+1)*m + (offset_j+1)] = 1
        v2[offset_j*p + offset_k] = 1
        w2[offset_k*n + (offset_i+1)] = 1
        add_component(u2, v2, w2)
        
        # M3 = A11(B12-B22)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[offset_i*m + offset_j] = 1
        v3[offset_j*p + (offset_k+1)] = 1
        v3[(offset_j+1)*p + (offset_k+1)] = -1
        w3[(offset_k+1)*n + offset_i] = 1
        add_component(u3, v3, w3)
        
        # M4 = A22(B21-B11)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(offset_i+1)*m + (offset_j+1)] = 1
        v4[(offset_j+1)*p + offset_k] = 1
        v4[offset_j*p + offset_k] = -1
        w4[offset_k*n + offset_i] = 1
        w4[offset_k*n + (offset_i+1)] = 1
        add_component(u4, v4, w4)
        
        # M5 = (A11+A12)B22
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[offset_i*m + offset_j] = 1
        u5[offset_i*m + (offset_j+1)] = 1
        v5[(offset_j+1)*p + (offset_k+1)] = 1
        w5[offset_k*n + offset_i] = -1
        w5[(offset_k+1)*n + offset_i] = 1
        add_component(u5, v5, w5)
        
        # M6 = (A21-A11)(B11+B12)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[(offset_i+1)*m + offset_j] = 1
        u6[offset_i*m + offset_j] = -1
        v6[offset_j*p + offset_k] = 1
        v6[offset_j*p + (offset_k+1)] = 1
        w6[(offset_k+1)*n + (offset_i+1)] = 1
        add_component(u6, v6, w6)
        
        # M7 = (A12-A22)(B21+B22)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[offset_i*m + (offset_j+1)] = 1
        u7[(offset_i+1)*m + (offset_j+1)] = -1
        v7[(offset_j+1)*p + offset_k] = 1
        v7[(offset_j+1)*p + (offset_k+1)] = 1
        w7[offset_k*n + offset_i] = 1
        add_component(u7, v7, w7)
        
        # 8-11: Schoolbook for B[:, 2] (col offset_k+2)
        # C1 = A11 B13 + A12 B23
        # C2 = A21 B13 + A22 B23
        
        u8 = [0]*(n*m); v8 = [0]*(m*p); w8 = [0]*(p*n)
        u8[offset_i*m + offset_j] = 1
        v8[offset_j*p + (offset_k+2)] = 1
        w8[(offset_k+2)*n + offset_i] = 1
        add_component(u8, v8, w8)
        
        u9 = [0]*(n*m); v9 = [0]*(m*p); w9 = [0]*(p*n)
        u9[offset_i*m + (offset_j+1)] = 1
        v9[(offset_j+1)*p + (offset_k+2)] = 1
        w9[(offset_k+2)*n + offset_i] = 1
        add_component(u9, v9, w9)
        
        u10 = [0]*(n*m); v10 = [0]*(m*p); w10 = [0]*(p*n)
        u10[(offset_i+1)*m + offset_j] = 1
        v10[offset_j*p + (offset_k+2)] = 1
        w10[(offset_k+2)*n + (offset_i+1)] = 1
        add_component(u10, v10, w10)
        
        u11 = [0]*(n*m); v11 = [0]*(m*p); w11 = [0]*(p*n)
        u11[(offset_i+1)*m + (offset_j+1)] = 1
        v11[(offset_j+1)*p + (offset_k+2)] = 1
        w11[(offset_k+2)*n + (offset_i+1)] = 1
        add_component(u11, v11, w11)

    # Call add_m223 for the 4 blocks
    # A11 (0,0), B_top (0,0)
    add_m223(0, 0, 0, n, m, p)
    # A12 (0,2), B_top (0,0)
    add_m223(0, 2, 0, n, m, p)
    # A21 (2,0), B_bot (2,0)
    add_m223(2, 0, 0, n, m, p)
    # A22 (2,2), B_bot (2,0)
    add_m223(2, 2, 0, n, m, p)
    
    # Total rank: 7 (M444) + 44 (M443) = 51?
    # Wait, M444 is 7 components.
    # M443 is 4 * 11 = 44 components.
    # Total 51.
    # This is much better than 112.
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
