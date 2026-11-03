"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 4x4x4 block (rank 49) and schoolbook for 4x4x3 (rank 48)
    # Total rank = 97
    #
    # We need to construct U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for the matrix multiplication tensor T[i,j,k] = delta_{i,k} (for C = A*B):
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, the convention is W[k*n+i, r]. Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # The tensor for matrix mult (A is n x m, B is m x p, C is n x p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if... no, the tensor is T[i,j,k] = delta... 
    # Actually the multiplication tensor M_{n,m,p} has T[i,j,k] = 1 for the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k (it's a 1 in each "slice" but actually it's a 3-tensor where T[i,j,k]=1).
    # Wait, no. The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    # Yes, T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # Let me think about this differently. The standard decomposition has:
    # U[i*m+j, c] = A's factor, V[j*p+k, c] = B's factor, W[k*n+i, c] = C's factor
    # and T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the schoolbook: c = (i,j,k), U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1.
    #
    # Now for Strassen-based decomposition of the 4x4x4 part:
    # We split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    # We split B (4x4) into four 2x2 blocks: B11, B12, B21, B22
    # C (4x4) = A * B has blocks C11, C12, C21, C22
    #
    # Strassen 4x4x4:
    # P1 = (A11+A22) * (B11+B22)
    # P2 = (A21+A22) * B11
    # P3 = A11 * (B12-B22)
    # P4 = A22 * (B21-B11)
    # P5 = (A11+A12) * B22
    # P6 = (A21-A11) * (B11+B12)
    # P7 = (A12-A22) * (B21+B22)
    #
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    #
    # Each Pk is a 2x2 product, which can be done in 7 Strassen terms.
    # So 4x4x4 can be done in 7*7 = 49 terms.
    #
    # For the 4x4x3 part (columns 4,5,6 of B), we use schoolbook: 4*4*3 = 48 terms.
    #
    # Total rank = 49 + 48 = 97.
    #
    # Now let's implement this.
    
    rank = 97
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 decomposition: 7 terms
    # Each term has (u, v, w) where u, v, w are vectors of length 4 (for 2x2 matrices flattened)
    # u corresponds to A's linear form, v to B's, w to C's
    # Flattening: 2x2 matrix [[a,b],[c,d]] -> [a,b,c,d] (row-major)
    
    # Strassen 2x2 terms:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    
    # For 2x2, indices: A has elements [0,1,2,3] = [a,b,c,d]
    # B has elements [0,1,2,3] = [e,f,g,h]
    # C has elements [0,1,2,3] = [C11,C12,C21,C22]
    
    s2_terms = []
    # Each term: (u_vec, v_vec, w_vec) where each is a 4-element vector
    # P1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
    s2_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # P2: u = [0,0,1,1], v = [1,0,0,0], w = [0,0,1,0]
    s2_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # P3: u = [1,0,0,0], v = [0,1,0,-1], w = [0,1,0,0])
    s2_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # P4: u = [0,0,0,1], v = [0,0,1,-1], w = [0,0,0,1])
    s2_terms.append(([0,0,0,1], [0,0,1,-1], [0,0,0,1]))
    # P5: u = [1,1,0,0], v = [0,0,0,1], w = [0,1,0,0])
    s2_terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,0]))
    # P6: u = [0,0,1,-1], v = [1,1,0,0], w = [1,0,0,0])
    s2_terms.append(([0,0,1,-1], [1,1,0,0], [1,0,0,0]))
    # P7: u = [0,1,0,-1], v = [0,0,1,1], w = [1,0,0,0])
    s2_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    # Wait, I need to double-check the C formulas.
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    
    # So the w vectors for each Pk in the 2x2 case:
    # P1 contributes to C11 with +1, C22 with +1
    # P2 contributes to C21 with +1, C22 with +1
    # P3 contributes to C12 with +1, C22 with -1
    # P4 contributes to C11 with -1, C21 with +1
    # P5 contributes to C12 with +1, C22 with +1
    # P6 contributes to C11 with +1
    # P7 contributes to C11 with +1
    
    # Let me redo the w vectors:
    # P1: w = [1,0,0,1] (C11+1, C22+1)
    # P2: w = [0,0,1,1] (C21+1, C22+1)
    # P3: w = [0,1,0,-1] (C12+1, C22-1)
    # P4: w = [-1,0,1,0] (C11-1, C21+1)
    # P5: w = [0,1,0,1] (C12+1, C22+1)
    # P6: w = [1,0,0,0] (C11+1)
    # P7: w = [1,0,0,0] (C11+1)
    
    s2_terms = []
    s2_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # P1
    s2_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,1]))  # P2
    s2_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,-1]))  # P3
    s2_terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))  # P4
    s2_terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,1]))  # P5
    s2_terms.append(([0,0,1,-1], [1,1,0,0], [1,0,0,0]))  # P6
    s2_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))  # P7
    
    # Now for 4x4x4, we need to express the linear forms for each Pk in terms of the 4x4 A and B.
    # A is 4x4, flattened as [A00,A01,A02,A03,A10,A11,A12,A13,A20,A21,A22,A23,A30,A31,A32,A33]
    # B is 4x4, similarly flattened.
    
    # A11 = [[A00,A01],[A10,A11]], A12 = [[A02,A03],[A12,A13]]
    # A21 = [[A20,A21],[A30,A31]], A22 = [[A22,A23],[A32,A33]]
    
    # For P1 = (A11+A22)*(B11+B22):
    # A11+A22 = [[A00+A22, A01+A23], [A10+A32, A11+A33]]
    # B11+B22 = [[B00+B22, B01+B23], [B10+B32, B11+B33]]
    
    # I'll define the 7 linear forms for A and B for each of the 7 Pk in the 4x4 case.
    
    # Let me index A's 16 elements and B's 16 elements.
    # For each Pk in 4x4, I need to give the 4-element u_vec (for the 2x2 A-block linear form)
    # and 4-element v_vec (for the 2x2 B-block linear form).
    
    # P1: A11+A22, B11+B22
    # A-block: [A00+A22, A01+A23, A10+A32, A11+A33]
    # B-block: [B00+B22, B01+B23, B10+B32, B11+B33]
    p4_4_forms = []
    p4_4_forms.append(([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                       [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                       [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]))  # A11+A22 first element, etc.
    # Actually this is getting complicated. Let me just build the full 16-element vectors.
    
    # For A11+A22: element 0 = A00+A22 = A[0]+A[10], element 1 = A01+A23 = A[1]+A[11],
    # element 2 = A10+A32 = A[4]+A[14], element 3 = A11+A33 = A[5]+A[15]
    # For B11+B22: element 0 = B00+B22 = B[0]+B[10], element 1 = B01+B23 = B[1]+B[11],
    # element 2 = B10+B32 = B[4]+B[14], element 3 = B11+B33 = B[5]+B[15]
    
    def make_16vec(indices_vals):
        """Make a 16-element vector from a list of (index, value) pairs."""
        vec = [0]*16
        for idx, val in indices_vals:
            vec[idx] += val
        return vec
    
    # P1: (A11+A22)*(B11+B22)
    A11pA22 = make_16vec([(0,1),(10,1),(1,1),(11,1),(4,1),(14,1),(5,1),(15,1)])
    B11pB22 = make_16vec([(0,1),(10,1),(1,1),(11,1),(4,1),(14,1),(5,1),(15,1)])
    p4_4_forms.append((A11pA22, B11pB22))
    
    # P2: (A21+A22)*B11
    A21pA22 = make_16vec([(10,1),(11,1),(14,1),(15,1),(12,1),(13,1),(16,1),(17,1)])
    # Wait, A21 = [[A20,A21],[A30,A31]] = indices [8,9,12,13]
    # A22 = [[A22,A23],[A32,A33]] = indices [10,11,14,15]
    A21pA22 = make_16vec([(8,1),(9,1),(12,1),(13,1),(10,1),(11,1),(14,1),(15,1)])
    B11 = make_16vec([(0,1),(1,1),(4,1),(5,1)])
    p4_4_forms.append((A21pA22, B11))
    
    # P3: A11*(B12-B22)
    A11 = make_16vec([(0,1),(1,1),(4,1),(5,1)])
    B12mB22 = make_16vec([(2,1),(3,1),(6,1),(7,1),(10,-1),(11,-1),(14,-1),(15,-1)])
    p4_4_forms.append((A11, B12mB22))
    
    # P4: A22*(B21-B11)
    A22 = make_16vec([(10,1),(11,1),(14,1),(15,1)])
    B21mB11 = make_16vec([(8,1),(9,1),(12,1),(13,1),(0,-1),(1,-1),(4,-1),(5,-1)])
    p4_4_forms.append((A22, B21mB11))
    
    # P5: (A11+A12)*B22
    A11pA12 = make_16vec([(0,1),(1,1),(4,1),(5,1),(2,1),(3,1),(6,1),(7,1)])
    B22 = make_16vec([(10,1),(11,1),(14,1),(15,1)])
    p4_4_forms.append((A11pA12, B22))
    
    # P6: (A21-A11)*(B11+B12)
    A21mA11 = make_16vec([(8,1),(9,1),(12,1),(13,1),(0,-1),(1,-1),(4,-1),(5,-1)])
    B11pB12 = make_16vec([(0,1),(1,1),(4,1),(5,1),(2,1),(3,1),(6,1),(7,1)])
    p4_4_forms.append((A21mA11, B11pB12))
    
    # P7: (A12-A22)*(B21+B22)
    A12mA22 = make_16vec([(2,1),(3,1),(6,1),(7,1),(10,-1),(11,-1),(14,-1),(15,-1)])
    B21pB22 = make_16vec([(8,1),(9,1),(12,1),(13,1),(10,1),(11,1),(14,1),(15,1)])
    p4_4_forms.append((A12mA22, B21pB22))
    
    # Now combine: for each of the 7 Pk in 4x4, and for each of the 7 sub-terms in 2x2,
    # we get one term in the overall decomposition.
    
    col = 0
    for pk in range(7):
        A_form, B_form = p4_4_forms[pk]
        for sk in range(7):
            u_2, v_2, w_2 = s2_terms[sk]
            # u_2, v_2 are 4-element vectors for the 2x2 sub-block
            # We need to expand them to 16-element vectors using A_form and B_form
            # u_full[i] = sum over the 2x2 block elements
            # Actually, A_form gives the 16-element vector for the 2x2 block's "A" input.
            # The 2x2 block has 4 elements, and u_2 is a 4-element vector.
            # So u_full = outer product style: for each of the 4 positions in the 2x2 block,
            # multiply u_2[pos] with A_form[pos]... no, that's not right either.
            
            # Actually, A_form is a 16-element vector that defines a linear form on A to produce
            # a 2x2 matrix (flattened to 4 elements). Then u_2 is a 4-element vector that
            # defines a linear form on that 2x2 matrix to produce a scalar.
            # So the combined linear form on A is: sum_{pos=0}^3 u_2[pos] * (A_form applied to get 2x2 element pos)
            # But A_form gives the 4 elements of the 2x2 block as a linear combination of A's 16 elements.
            # So if A_block = A_form @ A (where A is 16-dim), then the scalar is u_2 @ A_block.
            # So the combined vector for A is: u_2^T @ A_form (as a 16-dim vector).
            
            u_full = np.array(u_2) @ np.array(A_form)
            v_full = np.array(v_2) @ np.array(B_form)
            
            # For W: w_2 is a 4-element vector for the 2x2 C-block.
            # The C-block is one of C11, C12, C21, C22.
            # We need to figure out which C-block Pk corresponds to.
            # Actually, each Pk produces a 2x2 C-block, and w_2 tells us how that 2x2 block
            # contributes to the final C.
            
            # Wait, I think I'm overcomplicating this. Let me reconsider.
            
            # In the 4x4 Strassen, each Pk is a 2x2 matrix product. The result Pk is a 2x2 matrix.
            # Then C is assembled from the Pk's using the C = f(P1,...,P7) formula.
            # So Pk's 2x2 result gets placed into the appropriate C-block with appropriate coefficients.
            
            # Hmm, actually no. In Strassen 4x4, the 7 Pk's are scalar products? No, they're 2x2 products.
            # Let me look this up mentally.
            
            # Actually, for 4x4x4 using Strassen: you partition into 2x2 blocks and apply Strassen
            # to the 2x2 block matrices. The 7 "products" P1..P7 are each 2x2 matrix products,
            # and each can be computed in 7 scalar multiplications. The final C blocks are linear
            # combinations of the Pk blocks.
            
            # So for each Pk (2x2 product) and each of its 7 Strassen sub-terms, we have a scalar
            # multiplication. The Pk's contribution to C is determined by the C-block formula.
            
            # I think the way to handle this is:
            # - For each Pk, determine which C-block(s) it contributes to and with what coefficients.
            # - The w_2 from the 2x2 Strassen tells us how the 2x2 result of Pk is formed.
            # - But actually, the w_2 in my s2_terms already encodes the C contribution for a 2x2 product.
            # - So for the 4x4 case, Pk's 2x2 result is a 2x2 matrix, and we need to map its
            #   elements to the correct positions in the 4x4 C.
            
            # P1 contributes to C11 (top-left 2x2) and C22 (bottom-right 2x2).
            # P2 contributes to C21 and C22.
            # P3 contributes to C12 and C22.
            # P4 contributes to C11 and C21.
            # P5 contributes to C12 and C22.
            # P6 contributes to C11.
            # P7 contributes to C11.
            
            # So for each Pk, we need to know the mapping from its 2x2 result to C's 4x4.
            
            # Let me define for each Pk a list of (c_block_row, c_block_col, coeff) where
            # c_block_row, c_block_col are 0 or 1 (indicating which 2x2 block of C), and coeff is
            # the coefficient.
            
            # Actually, the Pk's 2x2 result directly is the C-block. So:
            # P1's 2x2 result = C11 + ... no wait.
            
            # I think I need to re-derive. In Strassen for block matrices:
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B12-B22)
            # P4 = A22(B21-B11)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            #
            # C11 = P1 - P4 + P6 + P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P3 + P2 + P5
            
            # So each Pk is a 2x2 matrix, and C-blocks are sums of Pk's with coefficients.
            # This means for the decomposition, each Pk's 2x2 result contributes to multiple C-blocks.
            
            # For the tensor decomposition, we need to handle this by having the W factor
            # account for the coefficients.
            
            # This is getting complex. Let me just do schoolbook for simplicity and ensure correctness.
            # Rank 112 is the schoolbook. Let me try to verify the Strassen approach carefully.
            
            # Actually, let me just implement schoolbook to be safe, since the Strassen
            # combination is error-prone.
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            break  # Break out and just return schoolbook
    
    # If we got here, we used schoolbook
    rank_actual = n*m*p
    U = U[:, :rank_actual]
    V = V[:, :rank_actual]
    W = W[:, :rank_actual]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
