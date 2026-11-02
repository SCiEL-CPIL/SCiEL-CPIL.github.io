import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # A is 4x5, B is 5x5, C is 4x5
    # W index: k*n + i (k in 0..4, i in 0..3)
    
    rank_total = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        nonlocal rank_total
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        rank_total += 1

    def add_strassen_2x2x2(offset_i, offset_j, offset_k):
        # Adds a Strassen decomposition for 2x2x2 at given offsets
        # A is 2x2, B is 2x2, C is 2x2
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to express this as sum of rank-1 tensors.
        # Each M is a product of two linear combinations.
        # We expand each M into rank-1 terms.
        
        # M1: (A11 + A22)(B11 + B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M2: (A21 + A22)B11 = A21B11 + A22B11
        # M3: A11(B12 - B22) = A11B12 - A11B22
        # M4: A22(B21 - B11) = A22B21 - A22B11
        # M5: (A11 + A12)B22 = A11B22 + A12B22
        # M6: (A21 - A11)(B11 + B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7: (A12 - A22)(B21 + B22) = A12B21 + A12B22 - A22B21 - A22B22
        
        # Coefficients for A_ij, B_jk, C_ik
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add 7 rank-1 terms? No, Strassen is 7 multiplications.
        # But each multiplication is a scalar.
        # The tensor decomposition has rank 7.
        # We add 7 terms.
        
        terms = [
            # (A_indices, B_indices, C_indices, coeff_A, coeff_B, coeff_C)
            # M1
            ([(offset_i+0, offset_j+0), (offset_i+1, offset_j+1)], 
             [(offset_j+0, offset_k+0), (offset_j+1, offset_k+1)], 
             [(offset_k+0, offset_i+0), (offset_k+0, offset_i+1), (offset_k+1, offset_i+0), (offset_k+1, offset_i+1)], 
             [1, 1], [1, 1], [1, 1, 1, 1]), # This is not rank 1.
        ]
        
        # Strassen decomposition is not a sum of 7 rank-1 tensors in the naive sense.
        # It is a sum of 7 terms of the form (linear_comb_A * linear_comb_B) * linear_comb_C.
        # Each term is a tensor of rank 1? No, it's a product of two vectors in A and B,
        # mapped to C.
        # The term is $u \otimes v \otimes w$ where $u, v$ are linear combos.
        # So we can just add 7 terms.
        
        # Term 1: (A11 + A22) * (B11 + B22) * (C11 + C12 + C21 + C22)
        # This is not a single rank-1 tensor.
        # The rank of the Strassen algorithm is 7.
        # This means there are 7 scalars $M_1..M_7$.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # The contribution to C is $M_1 (C_{11} + C_{12} + C_{21} + C_{22})$? No.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$.
        # So $M_1$ contributes to $C_{11}, C_{22}$?
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # So $M_1$ contributes to $C_{11}$ and $C_{22}$.
        # The tensor for $M_1$ is $(e_{11}+e_{22}) \otimes (e_{11}+e_{22}) \otimes (e_{11}+e_{22})$.
        # This is a rank-1 tensor in the space of $2 \times 2 \times 2$?
        # Yes, it's the outer product of three vectors.
        
        # We add 7 such terms.
        
        def add_strassen_term(a_vec, b_vec, c_vec):
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            for (i, j) in a_vec:
                u[i*m + j] = 1
            for (j, k) in b_vec:
                v[j*p + k] = 1
            for (k, i) in c_vec:
                w[k*n + i] = 1
            add_term(u, v, w)

        # M1
        add_strassen_term([(offset_i+0, offset_j+0), (offset_i+1, offset_j+1)], 
                          [(offset_j+0, offset_k+0), (offset_j+1, offset_k+1)], 
                          [(offset_k+0, offset_i+0), (offset_k+1, offset_i+1)])
        # M2
        add_strassen_term([(offset_i+1, offset_j+0), (offset_i+1, offset_j+1)], 
                          [(offset_j+0, offset_k+0)], 
                          [(offset_k+0, offset_i+1)])
        # M3
        add_strassen_term([(offset_i+0, offset_j+0)], 
                          [(offset_j+0, offset_k+1), (offset_j+1, offset_k+1)], 
                          [(offset_k+1, offset_i+0)])
        # M4
        add_strassen_term([(offset_i+1, offset_j+1)], 
                          [(offset_j+1, offset_k+0), (offset_j+0, offset_k+0)], 
                          [(offset_k+0, offset_i+1)])
        # M5
        add_strassen_term([(offset_i+0, offset_j+0), (offset_i+0, offset_j+1)], 
                          [(offset_j+1, offset_k+1)], 
                          [(offset_k+1, offset_i+0)])
        # M6
        add_strassen_term([(offset_i+1, offset_j+0), (offset_i+0, offset_j+0)], 
                          [(offset_j+0, offset_k+0), (offset_j+0, offset_k+1)], 
                          [(offset_k+1, offset_i+1)])
        # M7
        add_strassen_term([(offset_i+0, offset_j+1), (offset_i+1, offset_j+1)], 
                          [(offset_j+1, offset_k+0), (offset_j+1, offset_k+1)], 
                          [(offset_k+0, offset_i+0), (offset_k+1, offset_i+1)])

    def add_schoolbook_2x2x2(offset_i, offset_j, offset_k):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[(offset_i+i)*m + (offset_j+j)] = 1
                    v[(offset_j+j)*p + (offset_k+k)] = 1
                    w[(offset_k+k)*n + (offset_i+i)] = 1
                    add_term(u, v, w)

    # 1. A1 B11 (4x4x4) using Strassen-Strassen
    # We decompose 4x4 into four 2x2 blocks.
    # Apply Strassen to the 2x2 blocks.
    # This requires 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is rank 7.
    # Total rank 49.
    
    # We need to map the 2x2 Strassen to the 4x4 indices.
    # This is complex. We will just use schoolbook for 4x4x4 to be safe and simple, 
    # which is rank 64.
    # 64 + 16 + 20 = 100. No improvement.
    
    # Let's try to implement the 49-rank version.
    # It involves 7 calls to a 2x2x2 Strassen routine, but with indices mapped.
    # Actually, the 49-rank decomposition is just 7 independent 2x2x2 Strassen decompositions
    # applied to the appropriate sub-matrices?
    # No, Strassen mixes the blocks.
    # M1 = (A11 + A22)(B11 + B22).
    # A11 is top-left 2x2. A22 is bottom-right 2x2.
    # So M1 involves indices from all four quadrants.
    # We can just add the 7 terms for the 4x4 case directly.
    
    # Given time constraints, I will use the Schoolbook 4x4x4 (Rank 64) + Schoolbook 4x4x1 (16) + Schoolbook 4x1x5 (20).
    # Total 100.
    # This is not better than the seed.
    
    # I must find a better decomposition.
    # R(4, 4, 4) is 49.
    # I will implement the 49-rank decomposition for the 4x4x4 part.
    
    # The 7 terms for 4x4x4 Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each M is a 2x2 matrix product.
    # We decompose each M into rank 7 using Strassen 2x2x2.
    # But the inputs to M are sums/differences of 2x2 blocks.
    # This means the linear combinations in the Strassen 2x2x2 terms will involve indices from multiple quadrants.
    
    # I will write a helper that adds a Strassen 2x2x2 term given the specific index sets.
    
    # For M1:
    # A_part = A11 + A22. Indices: (0,0), (0,1), (1,0), (1,1) for A11? No.
    # A11 is rows 0-1, cols 0-1.
    # A22 is rows 2-3, cols 2-3.
    # So A_part indices: (0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3).
    # B_part indices: (0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3).
    # C_part indices: (0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3).
    
    # We call add_strassen_2x2x2 with these index sets?
    # No, add_strassen_2x2x2 assumes contiguous 2x2 blocks.
    # I will modify it to accept index lists.
    
    def add_strassen_term_custom(a_indices, b_indices, c_indices):
        u = [0] * (n*m)
        v = [0] * (m*p)
        w = [0] * (p*n)
        for (i, j) in a_indices:
            u[i*m + j] = 1
        for (j, k) in b_indices:
            v[j*p + k] = 1
        for (k, i) in c_indices:
            w[k*n + i] = 1
        add_term(u, v, w)

    # M1
    a1 = [(0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3)]
    b1 = [(0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3)]
    c1 = [(0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3)]
    # Strassen 2x2x2 on these "virtual" 2x2 blocks?
    # No, the product (A11+A22)(B11+B22) is a 2x2 matrix? No, it's a 4x4 matrix?
    # A11 is 2x2. A22 is 2x2. Sum is 2x2? No, they are in different positions.
    # (A11+A22) is not a 2x2 matrix. It's a 4x4 matrix with non-zeros in two blocks.
    # So we cannot use Strassen 2x2x2 directly.
    
    # We must use the full 4x4x4 decomposition.
    # Given the complexity, I will fall back to Schoolbook 100 if I can't implement 85 quickly.
    # But 85 is better.
    
    # I will implement the 49-rank 4x4x4 by expanding the 7 Strassen terms into rank-1 tensors.
    # Each term is a product of two 4x4 matrices (sparse) and a 4x4 matrix (sparse).
    # This is effectively 7 * 16 = 112 terms? No.
    # The rank is 49.
    
    # I will use the schoolbook for now to ensure correctness, and try to optimize if time permits.
    # Actually, 100 is the seed. I need to beat 100.
    # 85 is a solid target.
    
    # I will implement the 4x4x4 part using the 49-rank decomposition found in literature.
    # It consists of 49 rank-1 terms.
    
    # For simplicity and safety within time limit, I will use the Schoolbook for 4x4x4 (64)
    # and see if I can improve the other parts.
    # 64 + 16 + 20 = 100.
    
    # I will try to implement the 49-rank 4x4x4.
    
    # If I fail, I return 100.
    
    # Let's try to construct the 49-rank decomposition.
    # It is known that R(4,4,4) <= 49.
    # I will use a known list of 49 tuples.
    
    # Since I don't have the list memorized, I will use the recursive Strassen.
    # R(4,4,4) = 7 * R(2,2,2) = 49.
    # The 7 terms of Strassen for 4x4 are:
    # M1 = (A11+A22)(B11+B22)
    # ...
    # Each M is a product of two 4x4 matrices.
    # We can decompose each product into rank 7 using Strassen 2x2x2?
    # No, the matrices are 4x4.
    # We can decompose each 4x4x4 product into rank 49 using Strassen 4x4x4?
    # This is circular.
    
    # I will use the Schoolbook 100.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                add_term(u, v, w)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
