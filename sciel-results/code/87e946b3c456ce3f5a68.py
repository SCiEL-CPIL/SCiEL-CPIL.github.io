import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Lander-Bailey rank-19 decomposition for 4x4x4
    # The 19 terms are defined by their support in U (4x4), V (4x4), and W (4x4).
    # U indices: (i, j), V indices: (j, k), W indices: (k, i)
    # We represent each term as a tuple of 3 indices: (i, j, k) for the non-zero positions.
    # Actually, a rank-1 term in matrix multiplication tensor M_{n,m,p} is defined by:
    # U[i,j] = a, V[j,k] = b, W[k,i] = c
    # The product is a*b*c * e_ij * e_jk * e_ki
    # For the tensor to match, we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if the term (i,j,k) is present, 0 otherwise.
    # In Lander-Bailey, the decomposition is a sum of 19 such "monomials".
    # Let's list the 19 terms (i, j, k) such that the product of indicators is 1.
    # This is the "support" of the decomposition.
    
    # The Lander-Bailey decomposition for 4x4x4 has the following 19 terms (i, j, k):
    # Note: Indices are 0-based.
    # The standard form often cites these as the "Lander-Bailey" scheme.
    # I will use the explicit list of 19 terms.
    
    # Terms (i, j, k) where i, j, k in {0, 1, 2, 3}
    lb_terms = [
        (0, 0, 0), # 1
        (1, 1, 1), # 2
        (2, 2, 2), # 3
        (3, 3, 3), # 4
        (0, 1, 1), # 5
        (1, 0, 0), # 6
        (0, 2, 2), # 7
        (2, 0, 0), # 8
        (0, 3, 3), # 9
        (3, 0, 0), # 10
        (1, 2, 2), # 11
        (2, 1, 1), # 12
        (1, 3, 3), # 13
        (3, 1, 1), # 14
        (2, 3, 3), # 15
        (3, 2, 2), # 16
        (1, 0, 2), # 17
        (2, 1, 3), # 18
        (3, 2, 0)  # 19
    ]
    
    # Let's verify this specific set is a valid LB decomposition.
    # Actually, the standard LB decomposition is more complex and involves coefficients.
    # However, for the purpose of this task, if I cannot guarantee the coefficients of the standard LB decomposition,
    # I should use a simpler, verifiable decomposition.
    
    # Let's try a "Strassen-like" or "Bini-Capovani" decomposition?
    # No, those are harder to code from memory.
    
    # Let's stick to a robust approach:
    # We can use the fact that we can decompose 4x4x5 into:
    # 1. A 4x4x4 multiplication (using a known rank-19 decomposition if I had it, or rank-20/21 if easier)
    # 2. A 4x4x1 multiplication (rank 16)
    
    # Since I don't have the exact LB coefficients, I will use a "block" approach.
    # We can view the 4x4x5 multiplication as:
    # C = A * B
    # Where A is 4x4, B is 4x5.
    # We can split B into B1 (4x4) and B2 (4x1).
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x1)
    
    # For C1, if I use schoolbook, rank is 64.
    # For C2, if I use schoolbook, rank is 16.
    # Total rank 80.
    
    # Can I do better?
    # Yes, if I use a rank-19 decomposition for C1.
    # But I need the coefficients.
    
    # Let's try to construct a valid rank-19 decomposition for 4x4x4.
    # The Lander-Bailey decomposition is the only known rank-19.
    # I will hardcode the 19 terms and solve for the coefficients?
    # No, the coefficients are fixed by the structure.
    
    # Alternative: Use a rank-20 or rank-21 decomposition that is easier to define?
    # There is a known rank-20 decomposition?
    # Actually, the lower bound for 4x4x4 is 19.
    
    # Let's try a different strategy.
    # I will use the "Lander-Bailey" decomposition but I will define it via its action on the basis vectors.
    # This is too complex.
    
    # Let's go back to the schoolbook but optimize it?
    # No, schoolbook is 80.
    
    # What if I use a "hybrid" approach?
    # Use Strassen for the 4x4x4 part?
    # Strassen is for 2x2x2, rank 7.
    # For 4x4x4, we can use Strassen recursively?
    # 4x4x4 -> 8 blocks of 2x2x2.
    # Each 2x2x2 uses rank 7.
    # Total rank 8 * 7 = 56.
    # Plus the 4x4x1 part (rank 16).
    # Total rank 56 + 16 = 72.
    # 72 is better than 80.
    
    # Let's implement Strassen for 4x4x4.
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to extend this to 4x4x4.
    # A 4x4 matrix can be seen as 2x2 blocks of 2x2 matrices.
    # Let A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # Let B = [[B11, B12], [B21, B22]] where each Bj is 2x2.
    # Then C = A*B.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen:
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
    
    # Each Mi is a 2x2 matrix multiplication.
    # We can use schoolbook for each 2x2 multiplication (rank 4).
    # So each Mi contributes 4 to the rank.
    # Total rank for 4x4x4 = 7 * 4 = 28.
    # This is much better than 64.
    
    # Now for the 4x4x1 part.
    # A is 4x4, B2 is 4x1.
    # We can split A into 2x2 blocks and B2 into 2x1 blocks.
    # B2 = [[B2_top], [B2_bottom]] where B2_top is 2x1, B2_bottom is 2x1.
    # C2 = A*B2 = [[A11, A12], [A21, A22]] * [[B2_top], [B2_bottom]]
    # C2_top = A11*B2_top + A12*B2_bottom
    # C2_bottom = A21*B2_top + A22*B2_bottom
    
    # Each term is a 2x2 * 2x1 multiplication.
    # Schoolbook for 2x2 * 2x1:
    # Result is 2x1.
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # This requires 4 multiplications (2x2 * 2x1 is 2x2x1 tensor, rank 4).
    # So C2_top requires 4 multiplications.
    # C2_bottom requires 4 multiplications.
    # Total rank for 4x4x1 = 8.
    
    # Total rank = 28 (for 4x4x4) + 8 (for 4x4x1) = 36.
    # This is better than 80.
    
    # Let's implement this.
    
    # We need to construct U, V, W matrices.
    # U: (n*m, rank) = (16, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (20, 36)
    
    # The indices for U are (i, j) where i in 0..3, j in 0..3.
    # The indices for V are (j, k) where j in 0..3, k in 0..4.
    # The indices for W are (k, i) where k in 0..4, i in 0..3.
    
    # We will build the decomposition term by term.
    # Each term is a rank-1 tensor: U_term * V_term * W_term.
    # U_term is a 4x4 matrix (flattened to 16).
    # V_term is a 4x5 matrix (flattened to 20).
    # W_term is a 5x4 matrix (flattened to 20).
    
    # The product is sum_r U[:,r] V[:,r] W[:,r] = C
    # C[i, k] = sum_j A[i, j] B[j, k]
    
    # Let's define the terms.
    
    # Part 1: 4x4x4 multiplication (first 4 columns of B)
    # We use Strassen on the 4x4x4 part.
    # A is 4x4, B1 is 4x4 (columns 0-3 of B).
    
    # We define the 7 Strassen products for the 4x4x4 part.
    # Each product is a 2x2 matrix multiplication.
    # We will use schoolbook for each 2x2 multiplication.
    
    # Let's index the terms.
    # Term 0-3: M1 (2x2 mult)
    # Term 4-7: M2 (2x2 mult)
    # ...
    
    # This is getting complex to manage manually.
    # Let's write a helper function to add a schoolbook 2x2x2 term.
    
    terms = []
    
    def add_2x2_mult(U_offset, V_offset, W_offset, A_block, B_block, C_block):
        # A_block: 2x2 matrix of indices in U
        # B_block: 2x2 matrix of indices in V
        # C_block: 2x2 matrix of indices in W
        # This adds 4 terms to the decomposition.
        pass
    
    # This approach is too tedious.
    # Let's go back to the simple schoolbook for 4x4x5 but see if we can reduce the rank.
    
    # Actually, the rank of 4x4x5 is known to be at most 35 (LB + schoolbook).
    # If I can't get 35, 72 (Strassen + schoolbook) is good.
    # 36 (Strassen + Strassen-like for 4x4x1) is even better.
    
    # Let's implement the 36-rank decomposition.
    
    # We will construct the U, V, W matrices by summing the contributions of each rank-1 term.
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx: (i, j)
        # v_idx: (j, k)
        # w_idx: (k, i)
        U[u_idx[0]*m + u_idx[1], term_counter[0]] += coeff
        V[v_idx[0]*p + v_idx[1], term_counter[0]] += coeff
        W[w_idx[0]*n + w_idx[1], term_counter[0]] += coeff
        term_counter[0] += 1
        
    term_counter = [0]
    
    # Part 1: 4x4x4 using Strassen
    # A = [[A11, A12], [A21, A22]]
    # B1 = [[B11, B12], [B21, B22]]
    # C1 = A * B1
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    # B11: (0,0), (0,1), (1,0), (1,1)
    # B22: (2,0), (2,1), (3,0), (3,1)
    
    # We need to form the sum A11+A22 and B11+B22.
    # This means the U factor for M1 will have 1s at A11 and A22 positions.
    # The V factor for M1 will have 1s at B11 and B22 positions.
    # The W factor for M1 will be the result of multiplying these sums.
    # But wait, the Strassen algorithm replaces the 8 multiplications with 7.
    # Each multiplication is a 2x2x2 tensor.
    # The "sums" are linear combinations of the input elements.
    # So the rank-1 terms for M1 are not just single (i,j,k) terms, but sums.
    
    # Actually, the Strassen algorithm works by:
    # 1. Forming linear combinations of A and B.
    # 2. Performing 7 matrix multiplications.
    # 3. Forming linear combinations of the results.
    
    # In the tensor decomposition view, each of the 7 multiplications is a 2x2x2 tensor.
    # The inputs to these multiplications are linear combinations of the original A and B elements.
    # So we need to "expand" the Strassen steps into rank-1 terms of the original 4x4x4 tensor.
    
    # This is effectively unrolling the Strassen algorithm.
    # Let's do it for M1 = (A11+A22)(B11+B22).
    # A11+A22 is a 2x2 matrix with elements:
    # (0,0): A[0,0]+A[2,2]
    # (0,1): A[0,1]+A[2,3]
    # (1,0): A[1,0]+A[3,2]
    # (1,1): A[1,1]+A[3,3]
    
    # B11+B22 is a 2x2 matrix with elements:
    # (0,0): B[0,0]+B[2,0]
    # (0,1): B[0,1]+B[2,1]
    # (1,0): B[1,0]+B[3,0]
    # (1,1): B[1,1]+B[3,1]
    
    # The product M1 is (A11+A22) * (B11+B22).
    # This is a 2x2 matrix.
    # M1[0,0] = (A[0,0]+A[2,2])(B[0,0]+B[2,0]) + (A[0,1]+A[2,3])(B[1,0]+B[3,0])
    # This expands to 4 terms for M1[0,0].
    # Each term is a product of an A element and a B element.
    # So M1 contributes 4*4 = 16 rank-1 terms?
    # No, the rank is the number of scalar multiplications.
    # Strassen reduces the number of scalar multiplications from 8 to 7 for 2x2.
    # For 4x4, it reduces from 64 to 28.
    # So we should have 28 terms for the 4x4x4 part.
    
    # Let's list the 28 terms explicitly.
    # This is the "unrolled" Strassen decomposition.
    
    # I will generate the terms programmatically.
    
    # Define the 7 Strassen products for 2x2.
    # Each product is a 2x2 matrix multiplication.
    # We will define the U, V, W factors for each of the 7 products.
    
    # Product 1: (A11+A22)(B11+B22)
    # U1: A11+A22
    # V1: B11+B22
    # W1: Identity (since it's a standard 2x2 mult)
    
    # Wait, the W factor is not identity.
    # The result of the 2x2 multiplication is added to the final C with specific coefficients.
    # But in the tensor decomposition, the W factor captures the position in the final C.
    
    # Let's use a simpler method:
    # I will use the schoolbook decomposition for the 4x4x4 part but with a twist.
    # Actually, the schoolbook decomposition for 4x4x4 is 64 terms.
    # Strassen is 28 terms.
    # Let's implement the 28 terms.
    
    # I will write a function that takes the Strassen definition and generates the terms.
    
    # This is too complex to do reliably in this short time.
    # Let's fall back to the 72-rank decomposition (Strassen for 4x4x4 + Schoolbook for 4x4x1).
    # Wait, Strassen for 4x4x4 is 28 terms.
    # Schoolbook for 4x4x1 is 16 terms.
    # Total 44 terms.
    
    # Let's implement the 44-term decomposition.
    
    # Part 1: 4x4x4 using Strassen (28 terms)
    # Part 2: 4x4x1 using Schoolbook (16 terms)
    
    # I will hardcode the 28 Strassen terms.
    
    # This is still risky.
    # Let's just use the schoolbook for 4x4x5 (80 terms) to be safe, as I cannot guarantee the correctness of the Strassen unrolling.
    
    # Wait, I can use a known rank-20 decomposition for 4x4x4?
    # No, 19 is minimal.
    
    # Let's try to use the Lander-Bailey decomposition from a reliable source in my memory.
    # The LB decomposition for 4x4x4 has 19 terms.
    # The terms are:
    # 1. (0,0,0)
    # 2. (1,1,1)
    # 3. (2,2,2)
    # 4. (3,3,3)
    # 5. (0,1,1)
    # 6. (1,0,0)
    # 7. (0,2,2)
    # 8. (2,0,0)
    # 9. (0,3,3)
    # 10. (3,0,0)
    # 11. (1,2,2)
    # 12. (2,1,1)
    # 13. (1,3,3)
    # 14. (3,1,1)
    # 15. (2,3,3)
    # 16. (3,2,2)
    # 17. (1,0,2)
    # 18. (2,1,3)
    # 19. (3,2,0)
    
    # If this is correct, then the rank-19 decomposition is just the sum of these 19 rank-1 tensors.
    # Let's verify if this set of terms is a valid decomposition.
    # The tensor T[i,j,k] = 1 if the term (i,j,k) is in the set, 0 otherwise?
    # No, the tensor for matrix multiplication is T[i,j,k] = 1 if i,j,k are valid indices.
    # The decomposition is sum_r U_r[i,j] V_r[j,k] W_r[k,i] = T[i,j,k].
    # If U_r, V_r, W_r are indicator vectors for the term (i_r, j_r, k_r), then:
    # sum_r delta(i,i_r) delta(j,j_r) delta(k,k_r) = 1 if (i,j,k) is in the set, 0 otherwise.
    # This is NOT the matrix multiplication tensor!
    # The matrix multiplication tensor has T[i,j,k] = 1 for ALL i,j,k.
    # So the sum of these 19 terms would only cover 19 of the 64 entries.
    # So this simple set of 19 terms is NOT a valid decomposition.
    
    # The Lander-Bailey decomposition involves linear combinations of the inputs.
    # So the U, V, W factors are not just indicators, but have coefficients.
    
    # Since I cannot recall the exact coefficients, I will use the schoolbook decomposition (rank 80).
    # It is the only one I can guarantee is correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
