def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    1. Split the multiplication C = A * B into C_left = A * B_left (4x4 * 4x4) and C_right = A * B_right (4x4 * 4x1).
    2. Use two-level Strassen's algorithm for the 4x4 * 4x4 part.
       - Level 1: Split 4x4 into 2x2 blocks. 7 products of 2x2 matrices.
       - Level 2: Each 2x2 product uses Strassen (7 scalar multiplications).
       - Total rank for this part: 7 * 7 = 49.
    3. Use schoolbook multiplication for the 4x4 * 4x1 part.
       - Total rank for this part: 4 * 4 * 1 = 16.
    4. Total rank: 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coef=1):
        """
        Adds a rank-1 term to the decomposition.
        u_idx: index in U (i*m + j)
        v_idx: index in V (j*p + k)
        w_idx: index in W (k*n + i)
        coef: coefficient for this term
        """
        nonlocal col
        U[u_idx][col] += coef
        V[v_idx][col] += coef
        W[w_idx][col] += coef
        col += 1

    def strassen_2x2(a_indices, b_indices, c_indices):
        """
        Adds 7 terms for Strassen multiplication of two 2x2 matrices.
        a_indices, b_indices, c_indices: lists of 4 indices [top_left, top_right, bottom_left, bottom_right]
        """
        nonlocal col
        
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # M1 = (a00 + a11)(b00 + b11)
        # Contribution to C: c00 += M1, c11 -= M1, c01 -= M1, c10 -= M1
        add_term(a00, b00, c00, 1)
        add_term(a11, b00, c00, 1)
        add_term(a00, b11, c00, 1)
        add_term(a11, b11, c00, 1)
        # Wait, Strassen M1 is a single multiplication (a00+a11)*(b00+b11).
        # In the tensor decomposition, we represent this as sum of rank-1 terms.
        # (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11.
        # So we need 4 rank-1 terms for this one "virtual" multiplication if we expand it?
        # NO. The rank of the tensor is the number of scalar multiplications.
        # Strassen reduces 8 multiplications to 7.
        # The 7 multiplications are:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # Each Mi is a scalar multiplication of two linear combinations.
        # In the U, V, W representation, each scalar multiplication corresponds to ONE column in U, V, W.
        # The entries in U and V are the coefficients of the linear combinations.
        
        # Let's re-implement add_term to handle linear combinations.
        # Actually, the standard form is:
        # U[k, t] * V[k, t] * W[k, t]
        # where t is the rank index.
        
        # For M1 = (a00 + a11)(b00 + b11):
        # U entry for a00: 1, a11: 1
        # V entry for b00: 1, b11: 1
        # W entries: c00: 1, c11: -1, c01: -1, c10: -1
        # This is ONE rank-1 term in the decomposition sense?
        # No, the rank is the number of terms in the sum of products.
        # The term is (u1*a00 + u2*a11) * (v1*b00 + v2*b11) * (w1*c00 + w2*c11 + ...)?
        # No, the decomposition is:
        # C_ijk = sum_t U_ik,t V_jk,t W_kj,t (indices vary by convention)
        # Here we have C[i, k] = sum_j A[i, j] B[j, k].
        # Tensor T[i, j, k] = delta_ik? No, T is the multiplication tensor.
        # T[i, j, k] = 1 if i, j, k form a valid path? No.
        # T[i, j, k] is the coefficient of A[i,j]B[j,k] in C[i,k]. It is 1.
        # We want to write T as sum of rank-1 tensors U_t V_t W_t.
        # U_t is a vector in R^{n*m}, V_t in R^{m*p}, W_t in R^{p*n}.
        # (U_t V_t W_t)[i, j, k] = U_t[i, j] V_t[j, k] W_t[k, i].
        # Sum_t U_t[i, j] V_t[j, k] W_t[k, i] = 1 for all i, j, k?
        # No, the tensor T has T[i, j, k] = 1 if the indices match the multiplication structure.
        # Actually, T[i, j, k] = 1 if we are computing C[i, k] using A[i, j] and B[j, k].
        # So T[i, j, k] = 1 for all i, j, k? No, that's not right.
        # The tensor for matrix multiplication is defined such that:
        # C = A * B <=> <T, A \otimes B \otimes C> = 0? No.
        # Standard definition: T is the tensor such that C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}.
        # So T_{ijk} = 1 if the indices are consistent?
        # Actually, T_{ijk} = 1 for all i, j, k is not correct.
        # The tensor is T[i, j, k] = 1 if i, j, k are the indices of A, B, C respectively.
        # But C has indices i, k. A has i, j. B has j, k.
        # So the tensor is 3-way with dimensions n x m x p.
        # T[i, j, k] = 1.
        # We want to decompose T = sum_t U_t \otimes V_t \otimes W_t.
        # Where U_t has indices (i, j), V_t has indices (j, k), W_t has indices (k, i).
        # So (U_t \otimes V_t \otimes W_t)[i, j, k] = U_t[i, j] V_t[j, k] W_t[k, i].
        # We need sum_t U_t[i, j] V_t[j, k] W_t[k, i] = 1 for all i, j, k.
        
        # Let's check the schoolbook:
        # For each (i, j, k), we have a term with U[i, j]=1, V[j, k]=1, W[k, i]=1.
        # Sum over t (where t corresponds to i, j, k) gives 1 for that specific (i, j, k) and 0 for others.
        # So yes, rank is n*m*p.
        
        # Now for Strassen M1:
        # We are replacing 4 terms (a00*b00, a00*b11, a11*b00, a11*b11) with 1 term?
        # No, Strassen replaces 8 scalar multiplications with 7.
        # The 7 scalar multiplications are the "t" indices.
        # For M1, we have one scalar multiplication: m1 = (a00+a11)(b00+b11).
        # This contributes to C.
        # How does it contribute?
        # C00 += M1, C11 -= M1, C01 -= M1, C10 -= M1.
        # So for this single term t:
        # U_t[a00] = 1, U_t[a11] = 1
        # V_t[b00] = 1, V_t[b11] = 1
        # W_t[c00] = 1, W_t[c11] = -1, W_t[c01] = -1, W_t[c10] = -1
        # Let's verify if this single term works for the sub-tensor.
        # The sub-tensor for M1 is not just 1. It's a linear combination.
        # Actually, we are decomposing the identity tensor.
        # The Strassen algorithm provides a decomposition of the 2x2x2 multiplication tensor into 7 rank-1 tensors.
        
        # So, for each of the 7 Strassen steps, we add ONE column to U, V, W.
        
        # Reset col logic: I need to be careful. The `add_term` function I wrote above adds 1 to U, V, W.
        # But Strassen terms have multiple non-zero entries in U, V, W.
        # So I should NOT use `add_term` as defined (which assumes single non-zero U, V).
        # I need a function that sets multiple entries in U, V, W for a single column `col`.
        
        # Let's redefine the helper.
        
        # M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] -= 1
        W[c01][col] -= 1
        W[c10][col] -= 1
        col += 1
        
        # M2 = (a10 + a11)b00
        # C10 += M2, C11 -= M2
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a00(b01 - b11)
        # C01 += M3, C10 += M3
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        # C00 += M4, C10 += M4
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        # C00 -= M5, C11 += M5
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c11][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        # C11 += M6
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        # C00 += M7
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    # Define indices for 4x4 matrix
    # A[i, j] -> U[i*m + j] = U[i*4 + j]
    # B[j, k] -> V[j*p + k] = V[j*5 + k] (Note: for left part k is 0..3, for right part k is 4)
    # C[i, k] -> W[k*n + i] = W[k*4 + i]
    
    # Part 1: 4x4 * 4x4 (k=0..3)
    # Block indices (i, j) for A, (j, k) for B, (i, k) for C
    
    # A blocks
    A11 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # (0,0), (0,1), (1,0), (1,1)
    A12 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # (0,2), (0,3), (1,2), (1,3)
    A21 = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # (2,0), (2,1), (3,0), (3,1)
    A22 = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # (2,2), (2,3), (3,2), (3,3)
    
    # B blocks (k=0..3)
    B11 = [0*5+0, 0*5+1, 1*5+0, 1*5+1] # (0,0), (0,1), (1,0), (1,1)
    B12 = [0*5+2, 0*5+3, 1*5+2, 1*5+3] # (0,2), (0,3), (1,2), (1,3)
    B21 = [2*5+0, 2*5+1, 3*5+0, 3*5+1] # (2,0), (2,1), (3,0), (3,1)
    B22 = [2*5+2, 2*5+3, 3*5+2, 3*5+3] # (2,2), (2,3), (3,2), (3,3)
    
    # C blocks (k=0..3)
    C11 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # (0,0), (0,1), (1,0), (1,1)
    C12 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # (0,2), (0,3), (1,2), (1,3)
    C21 = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # (2,0), (2,1), (3,0), (3,1)
    C22 = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # (2,2), (2,3), (3,2), (3,3)
    
    # We need to map the "2x2 block" indices to the global indices.
    # The Strassen function expects lists of 4 indices.
    # For M1 = (A11 + A22)(B11 + B22), the "A" part is A11+A22.
    # This means the U entries for this term will have 1s at A11 indices and A22 indices.
    # My `strassen_2x2` function takes a list of 4 indices for A.
    # But A11+A22 is not a single 2x2 block of contiguous indices.
    # It is the sum of two blocks.
    # So I cannot just pass A11 to the function.
    # I need to modify `strassen_2x2` to accept linear combinations or handle the sums explicitly.
    
    # Let's create a helper to add a term where U is a sum of indices, V is a sum of indices, W is a sum of indices.
    
    def add_strassen_term(u_indices, v_indices, w_indices, w_coefs):
        """
        Adds one rank-1 term.
        u_indices: list of indices in U to add 1 to
        v_indices: list of indices in V to add 1 to
        w_indices: list of indices in W to add w_coefs[i] to
        """
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx, coef in zip(w_indices, w_coefs):
            W[idx][col] += coef
        col += 1
        
    # Now we can define the 7 terms for the 4x4*4x4 part.
    # M1 = (A11 + A22)(B11 + B22)
    # U: A11 + A22
    # V: B11 + B22
    # W: C11 + 1, C11' -1? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For M1:
    # W contributions: C11 += 1, C11 is the top-left 2x2 block of C.
    # Wait, C11 is a 2x2 block. M1 is a 2x2 matrix.
    # The formula C11 = M1 + ... means the 2x2 block C11 gets the 2x2 matrix M1 added to it.
    # So for each entry in C11, the coefficient is 1.
    # Similarly for other blocks.
    
    # So for M1:
    # U indices: A11 + A22 (8 indices)
    # V indices: B11 + B22 (8 indices)
    # W indices: C11 (4 indices, coef 1), C11? No.
    # The Strassen formula for 2x2 is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Here C00 is the top-left 2x2 block of the result.
    # So for M1, it contributes +1 to all 4 entries of C11 block?
    # No, M1 is a 2x2 matrix. C11 is a 2x2 matrix.
    # The equation is C11 = M1 + M4 - M5 + M7.
    # This means element-wise.
    # So M1 adds its 2x2 values to C11.
    # In the tensor decomposition, the "W" part for M1 should have 1s in the C11 block?
    # Let's check the 2x2 case.
    # If we use Strassen for 2x2, M1 is a scalar.
    # C00 (scalar) += M1.
    # So W[c00] += 1.
    # For 4x4, M1 is a 2x2 matrix.
    # C11 (2x2 matrix) += M1.
    # So W[c11_00] += 1, W[c11_01] += 1, W[c11_10] += 1, W[c11_11] += 1.
    # Yes.
    
    # So for M1:
    # U: A11 + A22
    # V: B11 + B22
    # W: C11 (all 4 entries) with coef 1.
    
    # Let's verify with the 2x2 Strassen derivation.
    # M1 = (a00+a11)(b00+b11)
    # C00 = M1 + ...
    # So W[c00] gets +1.
    # C11 = M1 - ...
    # So W[c11] gets +1?
    # Wait, in the 2x2 Strassen code I wrote earlier:
    # M1: W[c00] += 1, W[c11] -= 1, W[c01] -= 1, W[c10] -= 1.
    # Why?
    # Because C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    
    # So for M1:
    # C00 gets +1.
    # C11 gets +1.
    # C01 gets 0.
    # C10 gets 0.
    
    # My previous code:
    # W[c00][col] += 1
    # W[c11][col] -= 1
    # W[c01][col] -= 1
    # W[c10][col] -= 1
    
    # This implies C11 = ... - M1 ...?
    # Let's re-verify Strassen formulas.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # C00: +1
    # C11: +1
    # C01: 0
    # C10: 0
    
    # My previous code had:
    # W[c00] += 1
    # W[c11] -= 1
    # W[c01] -= 1
    # W[c10] -= 1
    
    # This is WRONG.
    # Let's check M2.
    # C10 = M2 + M4. So C10 gets +1 from M2.
    # C11 = M1 - M2 + ... So C11 gets -1 from M2.
    # My code:
    # W[c10] += 1
    # W[c11] -= 1
    # This matches M2.
    
    # Let's check M3.
    # C01 = M3 + M5. So C01 gets +1 from M3.
    # C10 = M2 + M4. So C10 gets 0 from M3?
    # Wait, C10 = M2 + M4. No M3.
    # My code:
    # W[c01] += 1
    # W[c10] += 1
    # This implies C10 gets +1 from M3. That is WRONG.
    
    # Let's re-read Strassen.
    # C01 = M3 + M5
    # C10 = M2 + M4
    # So M3 only affects C01.
    # My code for M3:
    # W[c01] += 1
    # W[c10] += 1
    # This is incorrect.
    
    # Let's re-derive or find a reliable source.
    # Wikipedia:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C11(+1), C22(+1)
    # M2: C21(+1), C22(-1)
    # M3: C12(+1), C22(+1)
    # M4: C11(+1), C21(+1)
    # M5: C11(-1), C12(+1)
    # M6: C22(+1)
    # M7: C11(+1)
    
    # Let's check my code against this.
    # M1: W[c00]+=1, W[c11]-=1... WRONG. Should be W[c00]+=1, W[c11]+=1.
    # M2: W[c10]+=1, W[c11]-=1. Matches C21(+1), C22(-1). Correct.
    # M3: W[c01]+=1, W[c10]+=1. WRONG. Should be W[c01]+=1, W[c11]+=1.
    # M4: W[c00]+=1, W[c10]+=1. Matches C11(+1), C21(+1). Correct.
    # M5: W[c00]-=1, W[c11]+=1. Matches C11(-1), C12(+1)? No, C12 is c01 in 2x2 notation?
    # In 2x2:
    # C00 (top-left) = C11 in block notation?
    # Let's stick to 2x2 indices:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # My code:
    # M1: c00(+1), c11(-1), c01(-1), c10(-1). WRONG.
    # M2: c10(+1), c11(-1). Correct.
    # M3: c01(+1), c10(+1). WRONG. Should be c01(+1), c11(+1).
    # M4: c00(+1), c10(+1). Correct.
    # M5: c00(-1), c11(+1). WRONG. Should be c00(-1), c01(+1).
    # M6: c11(+1). Correct.
    # M7: c00(+1). Correct.
    
    # Okay, I will use the corrected coefficients in the new implementation.
    
    # M1
    u_idx = A11 + A22
    v_idx = B11 + B22
    w_idx = C11 + C22
    w_coef = [1]*4 + [1]*4
    add_strassen_term(u_idx, v_idx, w_idx, w_coef)
    
    # M2
    u_idx = A21 + A22
    v_idx = B11
    w_idx = C21 + C22
    w_coef = [1]*4 + [-1]*4
    add_strassen_term(u_idx, v_idx, w_idx, w_coef)
    
    # M3
    u_idx = A11
    v_idx = B12 + B22
    # B12 - B22 means B12 indices get +1, B22 indices get -1
    v_coef = [1]*4 + [-1]*4
    # My add_strassen_term assumes V indices get +1.
    # I need to modify it to accept V coefficients or handle it separately.
    # Let's modify add_strassen_term to accept coefficients for V as well.
    
    # Re-defining add_strassen_term to be fully general.
    # I will just write the 7 terms explicitly in the code below to avoid confusion.
    
    # Reset col to 0 for Part 1
    col = 0
    
    # M1
    for idx in A11 + A22: U[idx][col] += 1
    for idx in B11 + B22: V[idx][col] += 1
    for idx in C11 + C22: W[idx][col] += 1
    col += 1
    
    # M2
    for idx in A21 + A22: U[idx][col] += 1
    for idx in B11: V[idx][col] += 1
    for idx in C21: W[idx][col] += 1
    for idx in C22: W[idx][col] -= 1
    col += 1
    
    # M3
    for idx in A11: U[idx][col] += 1
    for idx in B12: V[idx][col] += 1
    for idx in B22: V[idx][col] -= 1
    for idx in C12: W[idx][col] += 1
    for idx in C22: W[idx][col] += 1
    col += 1
    
    # M4
    for idx in A22: U[idx][col] += 1
    for idx in B21: V[idx][col] += 1
    for idx in B11: V[idx][col] -= 1
    for idx in C11: W[idx][col] += 1
    for idx in C21: W[idx][col] += 1
    col += 1
    
    # M5
    for idx in A11 + A12: U[idx][col] += 1
    for idx in B22: V[idx][col] += 1
    for idx in C11: W[idx][col] -= 1
    for idx in C12: W[idx][col] += 1
    col += 1
    
    # M6
    for idx in A21: U[idx][col] += 1
    for idx in A11: U[idx][col] -= 1
    for idx in B11 + B12: V[idx][col] += 1
    for idx in C22: W[idx][col] += 1
    col += 1
    
    # M7
    for idx in A12: U[idx][col] += 1
    for idx in A22: U[idx][col] -= 1
    for idx in B21 + B22: V[idx][col] += 1
    for idx in C11: W[idx][col] += 1
    col += 1
    
    # Part 2: 4x4 * 4x1 (k=4)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
